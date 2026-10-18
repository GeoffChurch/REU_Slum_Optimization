"""The arterial's displacement budget (`reblock.methods.arterial.stop`): the meter's share is the
footprint area destroyed (a touched building counted whole), the greedy stops before the step that
would start past `max_displacement` (so a stopped run is a commit prefix of a longer one), and every
run reports why it stopped. The fault (a deaf meter) is shown to break the stop."""
from __future__ import annotations

import math
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
from reblock.methods.arterial import (
    Access,
    Displacement,
    SnapToBoundary,
    Uncapped,
    engines,
)
from reblock.methods.arterial.primitives import _planarize
from reblock.methods.arterial.stop import DisplacementMeter, GreedyRun
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


def _run(block: Block, *, max_displacement: float, max_roads: int) -> GreedyRun:
    return engines._greedy_arterials(
        block, realizer=SnapToBoundary(lam=2.0), objective=Access(), cost=Displacement(),
        anchors=Uncapped(8), top_k=4, max_roads=max_roads, half_width_m=HW, workers=1,
        max_displacement=max_displacement)


def _destroyed(block: Block, roads: list[LineString]) -> float:
    c = block.buildings.displacement(road_corridor(_planarize(roads, block.crs, 2 * HW)))
    area = np.asarray(shapely.area(block.buildings.outlines.to_numpy()), dtype=np.float64)
    return float((area * (c > 0)).sum() / area.sum())


def test_meter_share_matches_destroyed_area() -> None:
    block = _dense_block()
    with _commits() as commits:
        _run(block, max_displacement=math.inf, max_roads=4)
    assert len(commits) >= 2
    meter = DisplacementMeter.of(block, HW)
    assert meter.share() == 0.0
    for k, road in enumerate(commits, start=1):
        meter.add(road)
        assert meter.share() == pytest.approx(_destroyed(block, commits[:k]), abs=1e-9)


def test_meter_counts_a_clipped_building_whole() -> None:
    # Two equal discs (radius NN / 2 = 10), a road clipping the first one's east edge: the share
    # is that building's whole area, half the total, where the fractional share is a sliver.
    pts = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(20.5, 0.5)], crs=UTM)
    block = _grid_block_with_points(pts)
    road = LineString([(10.0, -5.0), (10.0, 5.0)])
    meter = DisplacementMeter.of(block, HW)
    meter.add(road)
    c = block.buildings.displacement(road.buffer(HW))
    assert 0.0 < c[0] < 0.1 and c[1] == 0.0
    assert meter.share() == pytest.approx(0.5, abs=1e-12)


def test_displacement_stop_is_a_commit_prefix() -> None:
    block = _dense_block()
    with _commits() as commits:
        run = _run(block, max_displacement=math.inf, max_roads=6)
    assert run.commits == len(commits) >= 4
    d = _meter_after(block, commits[:3])
    assert _meter_after(block, commits[:2]) < d
    stopped = _run(block, max_displacement=d, max_roads=6)
    assert stopped.commits == 3 and stopped.stop == "max_displacement"
    want = _planarize(commits[:3], block.crs, 2 * HW)
    assert [g.wkt for g in stopped.roads.geometry] == [g.wkt for g in want.geometry]


def _meter_after(block: Block, roads: list[LineString]) -> float:
    meter = DisplacementMeter.of(block, HW)
    for r in roads:
        meter.add(r)
    return meter.share()


def test_displacement_stop_guard_sees_a_deaf_meter(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _dense_block()
    with _commits() as commits:
        _run(block, max_displacement=math.inf, max_roads=6)
    d = _meter_after(block, commits[:3])
    monkeypatch.setattr(DisplacementMeter, "add", lambda self, real: None)
    assert _run(block, max_displacement=d, max_roads=6).commits != 3


def test_stop_reasons() -> None:
    block = _dense_block()
    capped = _run(block, max_displacement=math.inf, max_roads=2)
    assert capped.stop == "max_roads" and capped.commits == 2
    small = _run(_grid_block(4), max_displacement=math.inf, max_roads=50)
    assert small.stop == "no_gain" and 0 < small.commits < 50


def test_a_run_with_no_gain_commits_nothing() -> None:
    # Every parcel of a 2 x 2 grid fronts a street.
    run = _run(_grid_block(2), max_displacement=math.inf, max_roads=5)
    assert run.stop == "no_gain" and run.commits == 0
    assert len(run.roads) == 0 and "drain" in run.roads.columns


def test_no_buildings_never_stops_on_displacement() -> None:
    block = _grid_block(5)                         # tests.block_fixtures.no_buildings
    assert len(block.buildings) == 0
    assert DisplacementMeter.of(block, HW).share() == 0.0
    run = _run(block, max_displacement=0.01, max_roads=3)
    assert run.stop in ("no_gain", "max_roads")


def test_max_displacement_splits_the_identity() -> None:
    assert replace(ARTERIAL, max_displacement=0.1).identity != replace(
        ARTERIAL, max_displacement=0.2).identity


def test_max_displacement_out_of_range_is_refused() -> None:
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            replace(ARTERIAL, max_displacement=bad)


def test_proposal_params_record_the_stop() -> None:
    p = replace(ARTERIAL, max_roads=2, workers=1).propose(_dense_block())
    assert p.params["stop"] in ("max_roads", "no_gain", "max_displacement")
    commits = p.params["commits"]
    assert isinstance(commits, int) and commits <= 2
    assert p.params["max_displacement"] == ARTERIAL.max_displacement
