"""The guard of the road methods' free-first stage (free_roads.FreeEnclosed) on real blocks, with
its fault injection (a patch in one process, no file edited): it passes on the code as it is and
must fail under its fault. It was transparency_check.py's `free-roads` (guard 3) until the
transparency experiment's code was retired with the raster (NOTES, "The complex as the runtime:
the raster retired"): then FREE's plan was held on each plan method's own graph and on its
claim's routed one (TransparentSubstrate: the same nodes and edges); the claims were the
experiment's, so the plan is now held on the plan methods' own graphs.

    S=research/roadless/free_roads_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S free-roads <ids,> [fault]   # fault: one-short

`free-roads`, on blocks with homes walled in at baseline: FREE's plan (the stage as the
transparency experiment configured it, the research lineup's, free_roads.lineup_free: area D
FREE_BUDGET under both weightings, the runtime's metric) on the routing graph of each plan method's preset (PLAN_METHODS: compare_config's
all_methods, common.arms(); clearance_looped's tree), the same roads on all; then the metric
itself (`_Oracle`: the gap complex's flow model, its reach on the fixed mesh with the roads carved,
independent of the stage's own test on the complex rebuilt) finds the same walled-in homes, each
held road frees its home (and every home its step says it frees) once the roads up to it are
carved, what the roads leave walled in is the plan's share, and their true displacement under both
weightings is the plan's and within its budget. Fault `one-short`: every held road one segment
short (its last point dropped as the stage routes it): the stage itself raises, its road not
freeing its home by the metric's test. Its blocks: 20269, 18895 and 23597 (1, 1 and 6 homes
walled in on the complex; 23597's home 272 found the targets' reach missing the form's growth:
NOTES, "Before the consolidated pass: three fixes").
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString

from reblock.budget import road_corridor
from reblock.contracts import Block, Method
from reblock.free_first import HeldRoad
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.cycle_native import CycleNativeReblocker
from reblock.methods.loop_closure import LoopClosureRefiner
from reblock.methods.resistance_lp import ResistanceLPReblocker
from reblock.methods.substrates import RoutingGraph
from reblock.permeability import with_width

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import free_roads as fr  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402

SINGLE = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
              NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
PLAN_METHODS = ("clearance_looped", "cycle_native", "resistance_lp")   # the methods with a stage
FREE = fr.lineup_free()  # the research lineup's stage (free_roads: its budget, margin and tie)
SHARE_TOL = 1e-12        # the oracle's walled-in share and displacements against the plan's


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _check_single_threaded() -> None:
    bad = {k: os.environ.get(k) for k, v in SINGLE.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"run single-threaded: set {bad} to {SINGLE}")


Tree = CycleNativeReblocker | ResistanceLPReblocker | ClearanceReblocker


def _tree(method: Method) -> Tree:
    """The method that holds roads first: clearance_looped's tree, or the method itself."""
    m = method.base if isinstance(method, LoopClosureRefiner) else method
    if not isinstance(m, CycleNativeReblocker | ResistanceLPReblocker | ClearanceReblocker):
        raise TypeError(f"{type(m).__name__} has no free-first stage")
    return m


def _method_graphs(b: Block) -> dict[str, tuple[RoutingGraph, float]]:
    """Per plan method, its preset's routing graph and road width."""
    presets = common.arms()
    out: dict[str, tuple[RoutingGraph, float]] = {}
    for name in PLAN_METHODS:
        tree = _tree(presets[name])
        out[name] = (tree.substrate.build(b), tree.road_width_m)
    return out


def _wkb_roads(p: fr.Plan) -> list[bytes]:
    return [bytes(w) for w in shapely.to_wkb(np.asarray([s.road.geometry for s in p.steps]))]


def _plans(b: Block, graphs: dict[str, tuple[RoutingGraph, float]]) -> dict[str, fr.Plan]:
    """FREE's plan on every graph; the held roads the same on all."""
    plans = {name: FREE.plan(b, graph, width_m=width) for name, (graph, width) in graphs.items()}
    first = next(iter(plans.values()))
    for name, p in plans.items():
        if (_wkb_roads(p) != _wkb_roads(first) or [s.home for s in p.steps]
                != [s.home for s in first.steps] or p.enclosed != first.enclosed):
            raise AssertionError(f"{b.block_id}: {name}'s graph holds other roads than "
                                 f"{next(iter(plans))}'s")
    return plans


class _Oracle:
    """B (i)'s test as the metric applies it, independent of the stage's own (free_roads._Reach,
    the complex rebuilt): the flow model's reach on the fixed mesh (gap_clear.Model under FREE's
    metric), the homes walled in at baseline its walled-in homes, a road set's the reach of the
    mesh with the roads' corridor carved into the footprints (gap_clear.Carving, one stop)."""

    def __init__(self, b: Block):
        self.b = b
        self.metric = gcl.complex_of(FREE.metric, factorization=gap_factor.SuperLU())
        m = self.metric.clearing(b, demand=common.PopulationDemand())
        self.enclosed, self.cost = m.walled0, m.cost

    def freed(self, roads: gpd.GeoDataFrame) -> NDArray[np.bool_]:
        """Per home: walled in at baseline and reached once `roads` are carved."""
        if not len(roads):
            return np.zeros(len(self.enclosed), dtype=bool)
        geo = gc.geometry(self.b)
        carving = gcl.Carving.of(geo, [[geo.local(road_corridor(roads))]])
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.metric.fixed(self.b, cache=Path(tmp), carving=carving)
        m = self.metric.model(self.b, fx, demand=common.PopulationDemand())
        live = m.live_homes_of(m.road_factors(gcl.Prefix(net=0, stop=1)))
        return np.asarray(self.enclosed & live)

    def enclosed_left(self, roads: gpd.GeoDataFrame) -> float:
        left = self.enclosed & ~self.freed(roads)
        return float(self.cost[left].sum())


def free_roads(ids: list[str]) -> list[str]:
    """The guard (the module's docstring): the blocks it fails on, with why."""
    _check_single_threaded()
    bad = []
    for b in common.build_blocks(ids):
        bid = b.block_id
        graphs = _method_graphs(b)
        try:
            p = next(iter(_plans(b, graphs).values()))
        except (RuntimeError, AssertionError) as e:
            bad.append(f"{bid}: the stage fails: {e}")
            print(f"{bid}: the stage fails: {e}", flush=True)
            continue
        width = next(iter(graphs.values()))[1]
        sc = _Oracle(b)
        if tuple(int(h) for h in np.flatnonzero(sc.enclosed)) != p.enclosed:
            bad.append(f"{bid}: the plan's walled-in homes {p.enclosed} are not the metric's "
                       f"{np.flatnonzero(sc.enclosed).tolist()}")
            continue
        if not p.enclosed:
            bad.append(f"{bid}: no home walled in at baseline: not a block for this guard")
            continue
        roads = [s.road.geometry for s in p.steps]

        def frame(k: int) -> gpd.GeoDataFrame:
            return with_width(gpd.GeoDataFrame(geometry=roads[:k], crs=b.crs), width)

        for k, s in enumerate(p.steps, start=1):
            freed = sc.freed(frame(k))
            if not freed[s.home] or not all(freed[j] for j in s.freed):
                bad.append(f"{bid}: road {k} (home {s.home}) does not free "
                           f"{[j for j in s.freed if not freed[j]] or [s.home]}")
        left = sc.enclosed_left(frame(len(roads)))
        polys = np.asarray(b.buildings.outlines)
        d_area = common.displacement(b, frame(len(roads)),
                                     common.AreaPopulation().weights(polys))
        d_count = common.displacement(b, frame(len(roads)),
                                      common.CountPopulation().weights(polys))
        if (abs(left - p.share_after) > SHARE_TOL or abs(d_area - p.D_area) > SHARE_TOL
                or abs(d_count - p.D_count) > SHARE_TOL or max(d_area, d_count) > FREE.budget):
            bad.append(f"{bid}: left {left} against {p.share_after}, D {d_area} {d_count} "
                       f"against {p.D_area} {p.D_count}")
        print(f"{bid}: {len(p.enclosed)} homes walled in ({', '.join(map(str, p.enclosed))}), "
              f"{len(p.steps)} roads held on the {len(graphs)} graphs ({', '.join(graphs)}), D "
              f"area {p.D_area:.4f} count {p.D_count:.4f}; each frees its home by the metric's "
              f"reach; left walled in {p.share_after:.5f} (the metric's {left:.5f})", flush=True)
    return bad


def _one_short(cheapest: Callable[..., HeldRoad | None]) -> Callable[..., HeldRoad | None]:
    """free_roads._cheapest with every road it returns one segment short (fault `one-short`)."""
    def short(*args: object, **kw: object) -> HeldRoad | None:
        road = cheapest(*args, **kw)
        if road is None or len(road.nodes) < 2:
            return road
        coords = list(road.geometry.coords)
        if len(coords) <= 2:
            return road
        return HeldRoad(geometry=LineString(coords[:-1]), nodes=road.nodes[:-1])
    return short


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {"free-roads": free_roads}
FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "free-roads": {"one-short": lambda: _patched(fr, "_cheapest", _one_short(fr._cheapest))},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    """gap_clear_check.py's convention: exit 0 when the guard passes, or fails under its fault."""
    if fault is None:
        bad = GUARDS[guard](ids)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard](ids)
    verdict = "FAILS on " + "; ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (2, 3) or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    main(a[0], common.ids_of(a[1]), a[2] if len(a) == 3 else None)
