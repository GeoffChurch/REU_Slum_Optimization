"""The arterial's real-block guards, each with its faults (a patch inside this process, no file
edited: free_roads_check.py's convention): a guard passes on the code as it is and must fail under
each of its faults.

    S=research/roadless/arterial_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S gain <ids,> <baseline-root> [fault]   # faults: always-live, no-min,
                                                           #   streets-only
    uv run python $S anchors <ids,> <baseline-root> [fault]  # faults: per-line
    uv run python $S meter <ids,> <baseline-root> [fault]  # faults: deaf, forgetful
    uv run python $S both-ways <guards,> <ids,> <baseline-root>  # each guard, then each fault

`gain`: the incremental access gain (objectives._AccessStep) against the full re-peel of the
committed union plus the candidate, on every chord the lineup's shortlist scores, at stored
committed networks. Per block, the lineup's stored ordered network
(<baseline-root>/scratch/networks/<id>/greedy_arterial_access_displacement.ordered.parquet:
canonical, street-first) cut at S in STEPS_S segments (capped at its length) is the committed
network (engines._step_network); that step's shortlist is built as engines._greedy_shortlist
builds it (the preset of road_rescore.lineup_arms()[ARTERIAL], FirstOrder(512, threads=1)); every
shortlisted chord realized and scored both ways. A mismatch fails unless it is a knife-edge FP
difference: every parcel whose seeding differs (depth 1 one way and not the other) lies within
KNIFE m of the tolerance from the seed geometry; those are counted and printed, and accepted.
Prints per S: candidates, non-live, gain > 0, accepted FP differences, and ms per candidate for
the reference and the incremental gain. Under a fault it stops at the first failure.
Faults: `always-live` (`_AccessStep.live` always True: a road that reaches no street seeds its
parcels), `no-min` (the relaxation's BFS assigns layer + 1 to every unvisited neighbour without
comparing it to the base depth), `streets-only` (liveness tested against the streets' union only,
committed roads ignored). Blocks: ZAF.9.3.1_1_20269 and ZAF.9.3.1_1_38616.

`anchors`: the samplers on 20269's stored ordered network cut at S in ANCHOR_S (streets plus the
first S segments): PerLineFallback(32, 128) gives the anchor counts the brief measured before
the samplers were moved out of primitives.py (ANCHORS_BEFORE: 112, 229, 323, 409, 525, from
arterial_rows/before/micro_20269.csv), so the move changed nothing, and HardCap(32, 128) gives
at most 128, sorted and unique, at every S. Fault `per-line`: anchors._global_samples replaced by
the per-line samples, so HardCap's cap fails to hold at S 607. Block: ZAF.9.3.1_1_20269.

`meter`: the access presets' stop (stop.Destroyed's incremental meter, its tau common.MEASURE's)
equals the lineup's scoring function (common.displacement under common.MEASURE, area weights) on
the same roads: the stored ordered network's segments fed to the meter one at a time and its D
read at S in METER_S (capped at the network's length), against the scoring function on the first
S segments, within METER_TOL. Faults: `deaf` (IncrementalDestroyed.add a no-op) and `forgetful`
(each piece replaces a building's covered region instead of joining it: shapely.union returns its
second argument). Blocks: any with a stored arterial network (ZAF.9.3.1_1_20269,
ZAF.9.3.1_1_38616).
"""
from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import road_rescore  # noqa: E402

from reblock.contracts import Block  # noqa: E402
from reblock.derive.access import (  # noqa: E402
    STREET_TOL,
    ParcelAdjacency,
    parcel_access_layers,
    past_every_parcel,
)
from reblock.destroyed import IncrementalDestroyed  # noqa: E402
from reblock.methods.arterial import anchors as anchors_mod  # noqa: E402
from reblock.methods.arterial import engines  # noqa: E402
from reblock.methods.arterial.anchors import AnchorSampler, HardCap, PerLineFallback  # noqa: E402
from reblock.methods.arterial.objectives import _AccessStep  # noqa: E402
from reblock.methods.arterial.primitives import (  # noqa: E402
    _candidate_chords,
    _deep_targets,
    _explode,
    _snap_graph,
)
from reblock.methods.arterial.realize import ChordRealizer  # noqa: E402
from reblock.methods.arterial.reblocker import GreedyArterialReblocker  # noqa: E402
from reblock.methods.arterial.shortlist import FirstOrder, RankContext  # noqa: E402
from reblock.methods.arterial.stop import Destroyed  # noqa: E402
from reblock.methods.boundary_graph import _boundary_graph  # noqa: E402

STEPS_S = (0, 50, 150, 300)    # committed segments of the stored network at which `gain` looks
KNIFE = 1e-9                   # m: a seeding difference this close to the tolerance is FP
SHORTLIST = FirstOrder(512, threads=1)
ANCHOR_S = (0, 150, 300, 450, 607)                       # `anchors`: the brief's S on 20269
ANCHORS_BEFORE = {0: 112, 150: 229, 300: 323, 450: 409, 607: 525}   # micro_20269.csv
CAPPED: tuple[AnchorSampler, AnchorSampler] = (PerLineFallback(32, 128), HardCap(32, 128))
METER_S = (1, 25, 100, 300)    # `meter`: segments of the stored network at which D is compared
METER_TOL = 1e-9


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _preset() -> GreedyArterialReblocker:
    m = road_rescore.lineup_arms()[road_rescore.ARTERIAL]
    assert isinstance(m, GreedyArterialReblocker)
    return m


def _ordered(root: Path, bid: str) -> list[LineString]:
    path = road_rescore.ordered_path(root / "scratch", bid, road_rescore.ARTERIAL)
    geoms = list(gpd.read_parquet(path).geometry)
    lines = [g for g in geoms if isinstance(g, LineString)]
    assert len(lines) == len(geoms), f"{path}: a road that is not a LineString"
    return lines


def _shortlist(m: GreedyArterialReblocker, block: Block, adjacency: ParcelAdjacency,
               committed: list[LineString], parcel_tree: STRtree, building_tree: STRtree
               ) -> list[LineString]:
    """The step's shortlisted chords, as engines._greedy_shortlist builds them."""
    hw = m.road_width_m / 2.0
    net = engines._step_network(committed, block, hw)
    curr = net.roads if committed else None
    network: list[BaseGeometry] = [*block.streets.geometry, *committed]
    chords = _candidate_chords(m.anchors.points(network), _deep_targets(adjacency, curr, m.top_k))
    depths = parcel_access_layers(adjacency, curr, unreached=past_every_parcel)
    return SHORTLIST.select(chords, RankContext(
        depths=depths.loc[block.parcels["parcel_id"]].to_numpy(dtype=float),
        parcel_tree=parcel_tree, building_tree=building_tree, half_width_m=hw,
        step=len(committed)))


def _knife_edge(step: _AccessStep, real: LineString, ref_depths: NDArray[np.int64],
                parcels: list[BaseGeometry]) -> bool:
    """Every parcel seeded one way and not the other lies within KNIFE of the tolerance from the
    trial's seed geometry (streets + committed + `real`), and at least one does."""
    inc_seeded = step.depths == 1
    if step.live(real):
        inc_seeded[step.seeded(real)] = True
    differ = np.flatnonzero(inc_seeded != (ref_depths == 1))
    if not len(differ):
        return False
    seed = unary_union([g for g in (step.seed, real) if g is not None])
    return all(abs(parcels[i].distance(seed) - step.tol) <= KNIFE for i in differ)


def gain(ids: list[str], root: Path) -> list[str]:
    """The guard `gain` (the module's docstring): the failures, each named."""
    road_rescore._check_single_threaded()
    m = _preset()
    hw = m.road_width_m / 2.0
    bad: list[str] = []
    blocks = {b.block_id: b for b in common.build_blocks(ids)}
    for bid in ids:
        block = blocks[bid]
        adjacency = ParcelAdjacency.of(block, STREET_TOL)
        scorer = m.objective.for_block(adjacency)
        sg = _snap_graph(_boundary_graph(block.parcels))
        parcels = list(block.parcels.geometry)
        parcel_tree = STRtree(parcels)
        building_tree = STRtree(shapely.points(block.buildings.xy))
        ordered = _ordered(root, bid)
        for s in sorted({min(s, len(ordered)) for s in STEPS_S}):
            committed = ordered[:s]
            net = engines._step_network(committed, block, hw)
            try:
                step = scorer.at_step(net, m.realizer)
            except AssertionError as e:
                bad.append(f"{bid} S {s}: {e}")
                print(f"{bid} S {s}: {e}", flush=True)
                return bad
            assert isinstance(step, _AccessStep)
            base = scorer.value(net.roads)
            merged = [net.merged] if net.merged is not None else []
            cands = _shortlist(m, block, adjacency, committed, parcel_tree, building_tree)
            n = dead = gaining = fp = 0
            t_ref = t_inc = 0.0
            for chord in cands:
                real = m.realizer.realize(chord, sg)
                if real is None or real.length == 0:
                    continue
                n += 1
                t = time.perf_counter()
                trial = _explode(unary_union([*merged, real]), block.crs, 2 * hw)
                want = scorer.value(trial) - base
                t_ref += time.perf_counter() - t
                t = time.perf_counter()
                got = step.of(real)
                t_inc += time.perf_counter() - t
                dead += not step.live(real)
                gaining += want > 0
                if got == want:
                    continue
                ref_depths = parcel_access_layers(adjacency, trial, unreached=past_every_parcel
                                                  ).to_numpy(dtype=np.int64)
                if _knife_edge(step, real, ref_depths, parcels):
                    fp += 1
                    print(f"  {bid} S {s}: accepted FP difference on {chord.wkt}: {got!r} "
                          f"against {want!r}", flush=True)
                    continue
                bad.append(f"{bid} S {s} {chord.wkt}: incremental {got!r} != re-peel {want!r}")
                print(bad[-1], flush=True)
                return bad
            print(f"{bid} S {s}: {n} candidates, {dead} not live, {gaining} with gain > 0, {fp} "
                  f"accepted FP differences; ms per candidate: re-peel "
                  f"{1e3 * t_ref / max(n, 1):.2f}, incremental {1e3 * t_inc / max(n, 1):.3f}",
                  flush=True)
    return bad


def anchors(ids: list[str], root: Path) -> list[str]:
    """The guard `anchors` (the module's docstring): the failures, each named."""
    road_rescore._check_single_threaded()
    per_line, hard = CAPPED
    assert isinstance(hard, HardCap)
    bad: list[str] = []
    blocks = {b.block_id: b for b in common.build_blocks(ids)}
    for bid in ids:
        ordered = _ordered(root, bid)
        streets: list[BaseGeometry] = list(blocks[bid].streets.geometry)
        for s in ANCHOR_S:
            if s > len(ordered):
                bad.append(f"{bid}: the stored network has {len(ordered)} segments, not {s}")
                continue
            network = [*streets, *ordered[:s]]
            n_per, pts = len(per_line.points(network)), hard.points(network)
            if bid == "ZAF.9.3.1_1_20269" and n_per != ANCHORS_BEFORE[s]:
                bad.append(f"{bid} S {s}: PerLineFallback gives {n_per} anchors, the brief's "
                           f"{ANCHORS_BEFORE[s]}")
            if len(pts) > hard.max_anchors or pts != sorted(set(pts)):
                bad.append(f"{bid} S {s}: HardCap gives {len(pts)} anchors (cap "
                           f"{hard.max_anchors}), sorted and unique: {pts == sorted(set(pts))}")
            print(f"{bid} S {s}: PerLineFallback {n_per}, HardCap {len(pts)}", flush=True)
    return bad


def meter(ids: list[str], root: Path) -> list[str]:
    """The guard `meter` (the module's docstring): the failures, each named."""
    road_rescore._check_single_threaded()
    hw = _preset().road_width_m / 2.0
    bad: list[str] = []
    blocks = {b.block_id: b for b in common.build_blocks(ids)}
    for bid in ids:
        b = blocks[bid]
        frame = gpd.read_parquet(road_rescore.ordered_path(root / "scratch", bid,
                                                           road_rescore.ARTERIAL))
        w = common.AreaPopulation().weights(np.asarray(b.buildings.outlines))
        m = Destroyed(share=1.0, tau=common.MEASURE.tau).start(b, hw)
        marks = sorted({min(s, len(frame)) for s in METER_S})
        for k, road in enumerate(frame.geometry, start=1):
            assert isinstance(road, LineString)
            m.add(road)
            if k not in marks:
                continue
            want = common.displacement(b, frame.iloc[:k], w, measure=common.MEASURE)
            got = m.share()
            print(f"{bid} S {k}: meter {got:.12f}, scoring {want:.12f}", flush=True)
            if abs(got - want) > METER_TOL:
                bad.append(f"{bid} S {k}: meter {got!r} against scoring {want!r}")
                return bad
    return bad


def _no_min(self: _AccessStep, seeds: NDArray[np.intp]) -> int:
    """Fault `no-min`: layer + 1 to every unvisited neighbour, never compared to the base depth."""
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


class _StreetsOnly:
    """Fault `streets-only`: liveness against the streets' union only. The streets are those of
    the block whose step is being scored: `_AccessStep` does not hold them, so the fault reads the
    block from the adjacency's last `at_step` (`_AccessBlock.at_step`, patched to record it)."""

    def __init__(self) -> None:
        self.streets: BaseGeometry | None = None

    def live(self, step: _AccessStep, real: LineString) -> bool:
        return self.streets is not None and bool(shapely.dwithin(real, self.streets, step.tol))


@contextmanager
def _streets_only() -> Iterator[None]:
    from reblock.methods.arterial import objectives
    fault = _StreetsOnly()
    at_step = objectives._AccessBlock.at_step

    def recording(self: objectives._AccessBlock, committed: objectives.CommittedNetwork,
                  realizer: ChordRealizer) -> objectives.StepGain:
        fault.streets = shapely.union_all(list(self.adjacency.block.streets.geometry))
        return at_step(self, committed, realizer)

    with (_patched(objectives._AccessBlock, "at_step", recording),
          _patched(_AccessStep, "live", lambda step, real: fault.live(step, real))):
        yield


def _always_live(self: _AccessStep, real: LineString) -> bool:
    del self, real
    return True


Guard = Callable[[list[str], Path], list[str]]
GUARDS: dict[str, Guard] = {"gain": gain, "anchors": anchors, "meter": meter}
FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "gain": {"always-live": lambda: _patched(_AccessStep, "live", _always_live),
             "no-min": lambda: _patched(_AccessStep, "relaxed_burden", _no_min),
             "streets-only": _streets_only},
    "anchors": {"per-line": lambda: _patched(anchors_mod, "_global_samples",
                                             anchors_mod._arclength_samples)},
    "meter": {"deaf": lambda: _patched(IncrementalDestroyed, "add", lambda self, piece: None),
              "forgetful": lambda: _patched(shapely, "union", lambda a, b: b)},
}


def main(guard: str, ids: list[str], root: Path, fault: str | None) -> None:
    """Exit 0 when the guard passes, or fails under its fault (free_roads_check.py's `main`)."""
    if fault is None:
        bad = GUARDS[guard](ids, root)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard](ids, root)
    verdict = "FAILS on " + "; ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


def both_ways(guards: list[str], ids: list[str], root: Path) -> None:
    """Each guard as it is, then under each of its faults, in this process (gap_clear_check.py's
    `both_ways`). Raises naming every run not as `main` expects."""
    wrong = []
    for guard in guards:
        for fault in (None, *FAULTS[guard]):
            try:
                main(guard, ids, root, fault)
            except SystemExit as e:
                wrong.append(f"{guard}{'' if fault is None else f' under {fault}'}: {e.code}")
    print(f"both ways, {', '.join(guards)} on {', '.join(ids)}: "
          + ("as expected" if not wrong else "NOT as expected: " + "; ".join(wrong)), flush=True)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == "both-ways" and set(a[1].split(",")) <= set(GUARDS):
        both_ways(a[1].split(","), common.ids_of(a[2]), Path(a[3]))
    elif len(a) in (3, 4) and a[0] in GUARDS:
        main(a[0], common.ids_of(a[1]), Path(a[2]), a[3] if len(a) == 4 else None)
    else:
        raise SystemExit(__doc__)
