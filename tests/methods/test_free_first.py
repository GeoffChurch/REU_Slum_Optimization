"""Free first in the road methods (`reblock.free_first`): roads held before a method builds are its
first committed roads -- leading its output and marked there, counted by its displacement
accounting, part of the network it routes from -- and holding nothing changes nothing.

The block is test_method_transparency's: 80 Voronoi parcels on a 90 m square with footprints that
straddle the parcel edges, streeted along the bottom. The held roads come from `Paths`, a stand-in
for the research strategy (research/roadless/free_roads.FreeEnclosed): the shortest paths on the
method's own graph from the street to fixed points at the back of the block.

Fault-injected where a docstring says so: the fault was applied, the test seen to fail, the code
restored.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace

import geopandas as gpd
import numpy as np
import pytest
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

from reblock.buildings import IncrementalOverlap
from reblock.contracts import Block, Method
from reblock.free_first import (
    HELD_COL,
    FreeFirst,
    HeldRoad,
    HoldNothing,
    held_count,
    mark_held,
)
from reblock.methods.cycle_native import CycleNativeReblocker
from reblock.methods.resistance_lp import ResistanceLPReblocker
from reblock.methods.substrates import RoutingGraph
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from tests.methods.test_clearance import CLEARANCE
from tests.methods.test_method_transparency import METHODS, SIDE_M, _block
from tests.test_loop_closure import LOOPS

BACK = ((0.5 * SIDE_M, 0.95 * SIDE_M), (0.15 * SIDE_M, 0.8 * SIDE_M))   # two points at the back


@dataclass(frozen=True)
class Paths:
    """Holds, in order, the shortest path (by the graph's own edge length) from the street to the
    node nearest each of `ends`, each street end first with its stub to the street."""

    ends: tuple[tuple[float, float], ...]

    @property
    def identity(self) -> Hashable:
        return ("paths", self.ends)

    @property
    def tag(self) -> str:
        return ":free-paths"

    def held(self, block: Block, graph: RoutingGraph, *,
             width_m: float) -> tuple[HeldRoad, ...]:
        del width_m
        street = unary_union(list(block.streets.geometry))
        seeds = np.flatnonzero(shapely.dwithin(shapely.points(graph.pts), street, graph.net_tol))
        n = len(graph.pts)
        csr = csr_matrix((graph.edist, (graph.rows, graph.cols)), shape=(n, n))
        _d, pred = dijkstra(csr, indices=seeds, min_only=True, return_predecessors=True)[:2]
        out = []
        for x, y in self.ends:
            path = [int(np.argmin(np.hypot(graph.pts[:, 0] - x, graph.pts[:, 1] - y)))]
            while pred[path[-1]] >= 0:
                path.append(int(pred[path[-1]]))
            path.reverse()
            coords = [tuple(graph.pts[k]) for k in path]
            sp = nearest_points(Point(coords[0]), street)[1]
            out.append(HeldRoad(geometry=LineString([(sp.x, sp.y), *coords]),
                                nodes=tuple(path)))
        return tuple(out)


ARMS = ("cycle_native", "resistance_lp", "clearance", "clearance_looped")


def _with(name: str, free: FreeFirst) -> Method:
    """The method `name` holding `free`'s roads first: clearance_looped's on its tree."""
    if name == "clearance_looped":
        return replace(LOOPS, base=replace(CLEARANCE, depth_target=1, free_first=free))
    m = METHODS[name]
    return replace(m, free_first=free)


def _roads(method: Method, block: Block) -> gpd.GeoDataFrame:
    roads = method.propose(block).roads
    assert roads is not None
    return roads


def _wkb(geoms: object) -> list[bytes]:
    return [bytes(w) for w in shapely.to_wkb(np.asarray(geoms))]


def _held_roads(block: Block, free: Paths) -> tuple[HeldRoad, ...]:
    graph = CLEARANCE.substrate.build(block)
    return free.held(block, graph, width_m=DEFAULT_ROAD_WIDTH_M)


# ------------------------------------------------------------- the strategy and the frame

def test_hold_nothing_holds_nothing_and_adds_nothing_to_a_label() -> None:
    block = _block()
    graph = CLEARANCE.substrate.build(block)
    assert HoldNothing().held(block, graph, width_m=DEFAULT_ROAD_WIDTH_M) == ()
    assert HoldNothing().tag == "" and HoldNothing().identity == ("HoldNothing", ())
    assert isinstance(HoldNothing(), FreeFirst) and isinstance(Paths(BACK), FreeFirst)


def test_a_frame_says_which_roads_are_held() -> None:
    """Nothing held: the frame as it was, no column. Something held: a leading block, read back.

    FAULT INJECTION: `mark_held` writing its column when nothing is held fails the first assert."""
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, i), (1, i)]) for i in range(4)])
    assert mark_held(roads, 0) is roads and held_count(roads) == 0
    marked = mark_held(roads, 2)
    assert marked[HELD_COL].tolist() == [True, True, False, False] and held_count(marked) == 2
    assert HELD_COL not in roads.columns
    with pytest.raises(ValueError):
        mark_held(roads, 5)


@pytest.mark.parametrize("col", [[False, True, True, False], [False] * 4,
                                 [True, False, True, True]])
def test_a_held_column_that_is_not_a_leading_block_is_refused(col: list[bool]) -> None:
    roads = gpd.GeoDataFrame({HELD_COL: col},
                             geometry=[LineString([(0, i), (1, i)]) for i in range(4)])
    with pytest.raises(ValueError):
        held_count(roads)


def test_a_held_column_must_be_bool() -> None:
    roads = gpd.GeoDataFrame({HELD_COL: [1, 1, 0]},
                             geometry=[LineString([(0, i), (1, i)]) for i in range(3)])
    with pytest.raises(TypeError):
        held_count(roads)


# ------------------------------------------------------------- the methods

@dataclass(frozen=True)
class Empty:
    """A strategy that holds nothing, but is not `HoldNothing`."""

    @property
    def identity(self) -> Hashable:
        return ("empty",)

    @property
    def tag(self) -> str:
        return ""

    def held(self, block: Block, graph: RoutingGraph, *,
             width_m: float) -> tuple[HeldRoad, ...]:
        del block, graph, width_m
        return ()


@pytest.mark.parametrize("name", ARMS)
def test_holding_nothing_is_todays_network_frame_and_all(name: str) -> None:
    """The methods' code with nothing held is the one they ran before: any strategy that holds
    nothing gives HoldNothing's frame, columns included. (The real check, against networks
    computed before free first existed, is transparency_check.py `hold-nothing`.)"""
    block = _block()
    a, b = _roads(_with(name, HoldNothing()), block), _roads(_with(name, Empty()), block)
    assert list(a.columns) == list(b.columns) == ["geometry", "width_m"]
    assert _wkb(a.geometry) == _wkb(b.geometry) and a.width_m.tolist() == b.width_m.tolist()


@pytest.mark.parametrize("name", ARMS)
def test_held_roads_lead_the_output_and_are_marked(name: str) -> None:
    """The held roads are the network's first rows, in their order, marked held; nothing after
    them is. clearance_looped's loop stage keeps its tree's marks.

    FAULT INJECTION: dropping `mark_held` from any method's output, or `held_count(base_roads)`
    from LoopClosureRefiner's (0 instead), fails that method's case; so does any method
    committing the held roads after its own first road."""
    block = _block()
    free = Paths(BACK)
    held = _held_roads(block, free)
    roads = _roads(_with(name, free), block)
    assert held_count(roads) == len(held) == 2 and len(roads) > len(held)
    assert _wkb(roads.geometry[:2]) == _wkb([r.geometry for r in held])
    assert roads[HELD_COL].tolist() == [True, True] + [False] * (len(roads) - 2)


def _true_displacement(block: Block, roads: list[BaseGeometry]) -> float:
    """Sum of c_b under the roads' corridor, the methods' own accounting (shapely buffers)."""
    tracker = IncrementalOverlap(block.buildings)
    tracker.add(unary_union([g.buffer(DEFAULT_ROAD_WIDTH_M / 2.0) for g in roads]))
    return tracker.total()


@pytest.mark.parametrize("name", ["cycle_native", "resistance_lp"])
def test_the_held_roads_spend_the_methods_displacement_cap(name: str) -> None:
    """The cap is on everything built, the held roads included: the network stays within it, and
    the method builds less of its own than without them.

    FAULT INJECTION: cycle_native not committing the held corridor to `taken` (the cap then reads
    only its cycles), and resistance_lp's `_round` handed its committed roads less the held ones
    (`best[len(held):]`), each fail their case: the network passes the cap by about the held
    roads' displacement."""
    block = _block()
    configured = METHODS[name]
    assert isinstance(configured, CycleNativeReblocker | ResistanceLPReblocker)
    cap = configured.max_displacement * len(block.buildings)
    held = list(_roads(_with(name, Paths(BACK)), block).geometry)
    plain = list(_roads(_with(name, HoldNothing()), block).geometry)
    assert _true_displacement(block, held) <= cap + 1e-9
    assert _true_displacement(block, plain) > _true_displacement(block, held[2:]) + 1e-9
    assert _true_displacement(block, held[:2]) > 0.0


def test_clearances_held_roads_serve_their_parcels_and_count_toward_max_roads() -> None:
    """clearance's tree grows from the held roads: with room for them alone it builds them and
    stops, and a road it adds can start on one, as on any road of the tree.

    FAULT INJECTION: `_drainage`'s `max_roads` blind to them (`roads` starting empty, the held
    geometries put first only when the greedy is done) fails the first assert; not adding their
    nodes to `net` fails the last."""
    block = _block()
    free = Paths(BACK)
    held = _held_roads(block, free)
    only = _roads(replace(CLEARANCE, free_first=free, max_roads=len(held)), block)
    assert len(only) == len(held) and held_count(only) == len(held)
    roads = _roads(replace(CLEARANCE, free_first=free), block)
    street = unary_union(list(block.streets.geometry))
    held_pts = shapely.points(np.concatenate([shapely.get_coordinates(r.geometry)
                                              for r in held]))
    off_street = held_pts[shapely.distance(held_pts, street) > 5.0]
    ends = shapely.points(np.asarray([shapely.get_coordinates(g)[-1]
                                      for g in list(roads.geometry)[len(held):]]))
    joined = shapely.STRtree(off_street).query(ends, predicate="dwithin", distance=1e-9)[0]
    assert len(joined) > 0, "no road of clearance's own starts on a held road"


@pytest.mark.parametrize("name", ARMS)
def test_a_held_run_is_labelled_apart(name: str) -> None:
    """A label names render files: holding roads adds the strategy's tag, holding nothing adds
    nothing.

    FAULT INJECTION: dropping `{self.free_first.tag}` from a method's `proposal_id` fails its
    case (clearance's fails clearance_looped's too)."""
    block = _block()
    plain = _with(name, HoldNothing()).propose(block).proposal_id
    held = _with(name, Paths(BACK)).propose(block).proposal_id
    assert ":free-paths" in held and held.replace(":free-paths", "") == plain


@pytest.mark.parametrize("name", ARMS)
def test_the_held_roads_are_in_the_identity(name: str) -> None:
    assert _with(name, Paths(BACK)).identity != _with(name, HoldNothing()).identity
    assert _with(name, Paths(BACK)).identity != _with(name, Paths(BACK[:1])).identity
