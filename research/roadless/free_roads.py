"""The road methods' free-first stage (BACKLOG 5; owner 2026-10-12, "Yes, let's try to free enclosed
homes first"): before a method builds anything, roads that free the homes enclosed at baseline,
as B's optimizer frees them first by clearing (gap_clear.Model.free_first). `FreeEnclosed` is a
`reblock.free_first.FreeFirst`: a method holding it commits these roads first and builds the rest
as it does today.

What it does, per block:
1. The homes enclosed at baseline, as the metric defines them: on the gap complex (`metric`, the
   runtime's: gap_clear.complex_of), a home is walled in when no ground within its ring reaches
   the street on the complex grown as the width form grows the footprints, read at the form's
   filtration width (gap_complex.walled: free first's test, which the flow model's reach agrees
   with home for home, gap_clear_check reach). A road frees it only by opening its own ground
   (gap_complex's HOMES: owner 2026-10-21, "No, its own pocket must open"), a groundless one's
   (no open ground within ring_m: gap_flow.groundless, by Placement's points on the complex) by
   any ground within ring_m.
2. For each, a road on the METHOD'S OWN routing graph (the one it built, handed in), from the
   street or a road already held, whose corridor, carved as the metric carves a road (Carve:
   exactly the corridor, every road buffered by half its width), opens ground that frees the
   home: the sealed pocket of the complex its own ground lies in (its components at the
   filtration width that hold a ring triangle and not the street); a groundless home's, ground
   within ring_m of its outline (its own footprint's included), or the sealed pocket of the
   complex those touch. Every edge and node of the graph within half the width of that region,
   less the width form's growth and `target_margin_m` (for the polygonal buffer), is a TARGET:
   the footprints a corridor cuts grow back into it by the form's growth (the shy distance's w /
   2), so only its ground that deep inside is open (23597's home 272: a corridor 0.80 m from it,
   its open ground 1.05 m, outside the 1 m ring; NOTES, "Before the consolidated pass: three
   fixes"). The road is the cheapest path to a target node, or through a target edge to its far
   end, in one Dijkstra a round, on the graph less every edge that runs into the metric's
   barriers (railways and water, open only at their crossings: a road across a stream frees
   nothing, the stream not a footprint; NOTES, "The consolidated pass").
3. The route's cost is DISPLACEMENT, not length: an edge weighs the population share its own
   corridor displaces (`population`'s weights, the share of each footprint the buffered edge
   covers), plus `length_tie` per metre so that among equal displacements the shorter road wins
   and no edge weighs 0. Summed along a path this over-counts where consecutive edges' corridors
   cover one building (their round joins), so it is a proxy; what the stage orders and budgets by
   is the TRUE displacement of the held roads' union corridor (common.displacement, as the
   prefixes measure it).
4. Order and budget as free_first: each round the home whose road adds the least true
   displacement per unit of its population (`population`) is held, while the held roads' true
   displacement stays within `budget` under both weightings the prefixes are cut by (count and
   area, road_rescore.POPS). Once held, the road must free its home by the metric's own test (the
   complex rebuilt with the held roads' corridor carved, gap_complex.build) or the stage raises;
   homes it frees in passing leave the queue.
Homes no target is reachable for are left enclosed and reported (`Plan.unreachable`), as are the
ones the budget leaves out (`Plan.over_budget`).

Why not route through free_first's crossing (gap_complex.free_first). The crossing is a set of
WHOLE buildings to clear; a road through it still pays for every building its 7 m corridor touches
on the way there, and nothing in the crossing makes that less. Weighting the substrate's own edges
by what their corridor displaces asks the road question directly, and needs no second graph: on
the ten held-out blocks with enclosed homes it freed every home on the raster (21252's home 631,
in a 558 m^2 parcel whose corners are 12 m from it, only through a parcel diagonal across its
pocket: hence the edge targets). Phase 2b moved its enclosure and its test from the raster to the
complex (NOTES, "The complex as the runtime: the ports and the cross-checks").

The identity covers what the roads are computed from beyond the block and the method's graph (each
in its own key): the fields (the metric's name among them, its code's digest in it), this module's
and its research imports' source (no reblock closure reaches them), and reblock.budget's closure
(road_corridor, the footprints' overlap).

The guard: free_roads_check.py `free-roads` (transparency_check.py's `hold-nothing`,
`no-enclosed` and `lead`, the transparency experiment's, went with its code).
"""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import gap_flow as gf  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402
from reblock.contracts import Block  # noqa: E402
from reblock.derive_graph import closure_hash, config_identity, source_hash  # noqa: E402
from reblock.destroyed import RampedDestroyed  # noqa: E402
from reblock.free_first import HeldRoad  # noqa: E402
from reblock.methods.substrates import RoutingGraph  # noqa: E402
from reblock.permeability import with_width  # noqa: E402

# This module's code and its research imports', in every identity: no reblock closure reaches them.
CODE = source_hash(Path(__file__), Path(common.__file__), Path(gc.__file__), Path(gf.__file__))
# The research lineup's stage (BACKLOG 5's TODO, made the road methods' default in the consolidated
# pass: road_rescore.lineup_arms) and the free-roads guard's (free_roads_check.FREE)
FREE_BUDGET = 0.05       # the stage's displacement budget (Lens A's: the transparency experiment's)
FREE_MARGIN_M = 0.05     # a target within half the road's width less the form's growth and this:
                         # the buffer's polygon falls inside its circle by under 0.005 m at a 3.5 m
                         # radius
FREE_LENGTH_TIE = 1e-12  # per metre on every edge: among equal displacements the shorter road
OBSTACLE_EDGE_M = 0.5    # a routing edge running deeper than this into a barrier is no edge (m:
                         # one along its edge, the street beside a railway, stays)
FREE_TRIES = 8           # a home whose this many roads (each to another target) each fail the
                         # metric's test is left walled in, `unfreed` (the targets near a barrier:
                         # 30796's home 3674, 45876's home 543)


class Population(Protocol):
    """A weighting of the homes (common.CountPopulation, common.AreaPopulation)."""

    @property
    def name(self) -> str: ...

    def weights(self, polys: NDArray[np.object_]) -> NDArray[np.float64]: ...


class Step(NamedTuple):
    """One held road: the home it was held for, the road, the true displacement it added (area-
    and count-weighted), and every enclosed home free once it is held (that home among them)."""

    home: int
    road: HeldRoad
    D_area: float
    D_count: float
    freed: tuple[int, ...]


class Displaced(NamedTuple):
    """True displacement under the two weightings the prefixes are cut by (road_rescore.POPS)."""

    by_area: float
    by_count: float


class Plan(NamedTuple):
    """A block's free-first stage: the homes enclosed at baseline, the held roads in order, the
    homes left enclosed (no reachable target, out of the budget, or FREE_TRIES roads that each
    failed the metric's test; `refused` the roads the test refused), the held roads' true
    displacement and the population share enclosed before and after (area-weighted)."""

    block_id: str
    enclosed: tuple[int, ...]
    steps: tuple[Step, ...]
    unreachable: tuple[int, ...]
    over_budget: tuple[int, ...]
    unfreed: tuple[int, ...]
    refused: int
    D_area: float
    D_count: float
    share_before: float
    share_after: float


class _Graph(NamedTuple):
    """The method's routing graph as this stage routes on it: one entry per undirected edge (the
    COO's rows < cols), each edge's displacement weight, the symmetric matrix of them, and per
    node whether it lies inside the metric's barriers (`blocked`: no road starts or ends there)."""

    ui: NDArray[np.int64]
    uj: NDArray[np.int64]
    segs: NDArray[np.object_]
    weight: NDArray[np.float64]
    matrix: csr_matrix
    blocked: NDArray[np.bool_]


class _Targets(NamedTuple):
    """One home's targets: the nodes and the undirected edges (into `_Graph`'s order) whose
    corridor frees it."""

    nodes: NDArray[np.intp]
    edges: NDArray[np.intp]


@dataclass(frozen=True, kw_only=True)
class FreeEnclosed:
    """Roads that free the homes enclosed at baseline (the module's docstring), within `budget` of
    true displacement under both weightings, ordered and routed by `population`'s; under `metric`,
    a gap complex metric's name (gap_clear.complex_of: its Spec and width form give the homes it
    walls in and its test of reach); a target within half the road's width less the form's
    growth and `target_margin_m` of the ground that frees a home; `length_tie` per metre on every
    edge."""

    budget: float
    metric: str
    population: Population
    target_margin_m: float
    length_tie: float
    measure: RampedDestroyed     # the true displacement's rule (common.MEASURE in the lineup)

    def __post_init__(self) -> None:
        if not 0.0 < self.budget <= 1.0:
            raise ValueError(f"a displacement budget lies in (0, 1]; got {self.budget}")
        if self.target_margin_m < 0.0 or self.length_tie <= 0.0:
            raise ValueError(f"target_margin_m >= 0 and length_tie > 0: {self.target_margin_m}, "
                             f"{self.length_tie}")
        gcl.complex_of(self.metric, factorization=gap_factor.SuperLU())    # the name is one

    @property
    def identity(self) -> Hashable:
        return ("FreeEnclosed", CODE, closure_hash("reblock.budget"),
                closure_hash("reblock.destroyed"), config_identity(self))

    @property
    def tag(self) -> str:
        return ":free-" + hashlib.sha256(repr(self.identity).encode()).hexdigest()[:8]

    def held(self, block: Block, graph: RoutingGraph, *,
             width_m: float) -> tuple[HeldRoad, ...]:
        return tuple(s.road for s in self.plan(block, graph, width_m=width_m).steps)

    def plan(self, block: Block, graph: RoutingGraph, *, width_m: float) -> Plan:
        """The stage on `block`, its roads on `graph` at `width_m` (the module's docstring)."""
        bid = block.block_id
        polys = np.asarray(block.buildings.outlines)
        metric = gcl.complex_of(self.metric, factorization=gap_factor.SuperLU())
        # the block as the metric meshes it: its barriers cut out (its scorer's reach, which the
        # leaks and the boost do not change)
        geo = gc.geometry(metric.exits.barriers.block(block))
        reach0, own = _Reach.baseline(geo, metric)
        enclosed = np.flatnonzero(~reach0.live)
        if not len(enclosed):
            return Plan(block_id=bid, enclosed=(), steps=(), unreachable=(), over_budget=(),
                        unfreed=(), refused=0, D_area=0.0, D_count=0.0, share_before=0.0,
                        share_after=0.0)
        w = self.population.weights(polys)                    # the key's and the route's
        area_w = common.AreaPopulation().weights(polys)         # the budget's two weightings
        count_w = common.CountPopulation().weights(polys)
        g = _routing(block, graph, width_m, w, self.length_tie,
                     metric.exits.barriers.obstacle(block))
        reach = width_m / 2.0 - metric.form.grow - self.target_margin_m
        targets = _targets(geo, reach0, own, graph, g, enclosed, reach, metric.form.filtration_w)
        street = unary_union(list(block.streets.geometry))
        seeds = np.flatnonzero(shapely.dwithin(shapely.points(graph.pts), street, graph.net_tol)
                               & ~g.blocked)
        is_seed = np.zeros(len(graph.pts), dtype=bool)
        is_seed[seeds] = True

        def frame(roads: list[HeldRoad]) -> gpd.GeoDataFrame:
            return with_width(gpd.GeoDataFrame(geometry=[r.geometry for r in roads],
                                               crs=block.crs), width_m)

        def displaced(roads: list[HeldRoad]) -> tuple[Displaced, float]:
            """The true displacement of `roads` under the budget's weightings, and under the
            key's (`population`)."""
            f = frame(roads)
            m = self.measure
            return (Displaced(by_area=common.displacement(block, f, area_w, measure=m),
                              by_count=common.displacement(block, f, count_w, measure=m)),
                    common.displacement(block, f, w, measure=m))

        held: list[HeldRoad] = []
        steps: list[Step] = []
        now, now_key = Displaced(by_area=0.0, by_count=0.0), 0.0
        left = ~reach0.live
        routed: set[int] = set()            # the homes the last round found a road for
        failed: dict[int, set[int]] = {}    # per home, the targets whose road the test refused
        tries: dict[int, int] = {}          # ... and how many of its roads it refused
        unfreed: set[int] = set()           # ... the homes FREE_TRIES of whose roads it refused
        while left.any():
            sources = np.unique(np.concatenate([seeds, *[np.asarray(r.nodes) for r in held]]))
            if not len(sources):
                routed = set()
                break
            dist, pred = dijkstra(g.matrix, indices=sources, min_only=True,
                                  return_predecessors=True)[:2]
            best: tuple[int, HeldRoad, int, Displaced, float] | None = None
            key = np.inf
            routed = set()
            for h in np.flatnonzero(left):
                if int(h) in unfreed:
                    continue
                got = _cheapest(graph, g, targets[int(h)], dist, pred, is_seed, street,
                                skip=failed.get(int(h), set()))
                if got is None:
                    continue
                road, target = got
                routed.add(int(h))
                after, after_key = displaced([*held, road])
                k = (after_key - now_key) / w[h]
                if after.by_area <= self.budget and after.by_count <= self.budget and k < key:
                    best, key = (int(h), road, target, after, after_key), k
            if best is None:
                break
            h, road, target, after, after_key = best
            held.append(road)
            freed = _Reach.of(geo, metric, own,
                              carve=geo.local(road_corridor(frame(held)))).live
            if not freed[h]:
                # the targets are the metric's reach read off the graph, near a barrier not
                # always its test: the road dropped, the home tries its next target
                held.pop()
                failed.setdefault(h, set()).add(target)
                tries[h] = tries.get(h, 0) + 1
                if tries[h] >= FREE_TRIES:
                    unfreed.add(h)
                print(f"{bid} free roads: the road for home {h} ({len(road.nodes)} nodes, "
                      f"{road.geometry.length:.1f} m) does not free it by the metric's test: "
                      + ("left walled in" if h in unfreed else "its next target"), flush=True)
                continue
            also = [int(j) for j in np.flatnonzero(left & freed) if j != h]
            steps.append(Step(home=h, road=road, D_area=after.by_area - now.by_area,
                              D_count=after.by_count - now.by_count,
                              freed=tuple(int(j) for j in np.flatnonzero(left & freed))))
            print(f"{bid} free roads: home {h} ({area_w[h] / area_w.sum():.5f} of the area "
                  f"population), a road of {len(road.nodes)} nodes and {road.geometry.length:.1f} "
                  f"m, D area {steps[-1].D_area:.5f} count {steps[-1].D_count:.5f}"
                  + (f", freeing {also} too" if also else ""), flush=True)
            now, now_key = after, after_key
            left &= ~freed
        unreachable = tuple(int(h) for h in np.flatnonzero(left)
                            if int(h) not in routed and int(h) not in unfreed)
        over = tuple(int(h) for h in np.flatnonzero(left) if int(h) in routed)
        plan = Plan(block_id=bid, enclosed=tuple(int(h) for h in enclosed), steps=tuple(steps),
                    unreachable=unreachable, over_budget=over,
                    unfreed=tuple(sorted(h for h in unfreed if left[h])),
                    refused=sum(tries.values()), D_area=now.by_area,
                    D_count=now.by_count,
                    share_before=float(area_w[~reach0.live].sum() / area_w.sum()),
                    share_after=float(area_w[left].sum() / area_w.sum()))
        print(f"{bid} free roads: {len(enclosed) - int(left.sum())} of {len(enclosed)} enclosed "
              f"homes freed by {len(steps)} roads, D area {plan.D_area:.5f} count "
              f"{plan.D_count:.5f}; left enclosed: unreachable {list(unreachable)}, over the "
              f"budget {list(over)}, unfreed {list(plan.unfreed)} ({plan.refused} roads refused)",
              flush=True)
        return plan


class _Reach(NamedTuple):
    """The metric's reach on the complex with a corridor carved (gap_complex.build on the
    footprints less `carve`, grown as the width form grows them): the complex, each home's ground
    that frees it (gap_complex.homes_own), and whether each home reaches the street at the form's
    filtration width."""
    cx: gc.Complex
    homes: gc.Homes
    live: NDArray[np.bool_]

    @staticmethod
    def baseline(geo: gc.Geometry, metric: gcl.ComplexMetric) -> tuple[_Reach, gc.Own]:
        """The reach with nothing carved (each home's ground its ring: its own ground then), and
        each home's own ground (the groundless among the homes walled in, by Placement's points
        on the complex: gap_flow.groundless)."""
        cx = gc.build(geo, metric.spec, np.ones(len(geo.polys), dtype=bool), carve=None,
                      grow=metric.form.grow)
        hm = gc.homes_of(geo, cx)
        live = np.asarray(~gc.walled(gc.home_exits(hm, gc.exit_widths(cx)),
                                     metric.form.filtration_w))
        own = gc.own_of(geo, cx, gf.groundless(geo, metric.spec.ring_m, metric.fs.quad, cx.xy,
                                               cx.tri, hm.ring, ~live))
        return _Reach(cx=cx, homes=hm, live=live), own

    @staticmethod
    def of(geo: gc.Geometry, metric: gcl.ComplexMetric, own: gc.Own, *,
           carve: BaseGeometry) -> _Reach:
        cx = gc.build(geo, metric.spec, np.ones(len(geo.polys), dtype=bool), carve=carve,
                      grow=metric.form.grow)
        hm = gc.homes_own(geo, own, cx)
        live = ~gc.walled(gc.home_exits(hm, gc.exit_widths(cx)), metric.form.filtration_w)
        return _Reach(cx=cx, homes=hm, live=np.asarray(live))


def _routing(block: Block, graph: RoutingGraph, width_m: float, w: NDArray[np.float64],
             length_tie: float, obstacle: BaseGeometry) -> _Graph:
    """Each undirected edge weighs the `w`-weighted share its own corridor displaces (the
    footprint share the edge buffered by width_m / 2 covers, as Extents.displacement measures a
    corridor) plus `length_tie` per metre; an edge that runs into the metric's barriers
    (`obstacle`, the block's CRS: deeper than OBSTACLE_EDGE_M inside it) is no edge and a node
    inside them no seed or target, a road crossing a railway or water only at its crossings
    (1182's home 180: its road across a stream freed nothing, the stream not a footprint; 30848's
    home 1506: a road of one street node inside the railway's strip)."""
    mask = graph.rows < graph.cols
    ui, uj = graph.rows[mask], graph.cols[mask]
    segs = np.asarray(shapely.linestrings(np.stack([graph.pts[ui], graph.pts[uj]], axis=1)),
                      dtype=object)
    blocked = np.zeros(len(graph.pts), dtype=bool)
    if not shapely.is_empty(obstacle):
        core = shapely.buffer(obstacle, -OBSTACLE_EDGE_M)
        shapely.prepare(core)
        open_ = ~np.asarray(shapely.intersects(core, segs))
        ui, uj, segs = ui[open_], uj[open_], segs[open_]
        blocked = np.asarray(shapely.contains(core, shapely.points(graph.pts)))
    bufs = np.asarray(gpd.GeoSeries(segs).buffer(width_m / 2.0), dtype=object)
    outlines = np.asarray(block.buildings.outlines, dtype=object)
    area = np.asarray(shapely.area(outlines), dtype=np.float64)
    e, b = shapely.STRtree(outlines).query(bufs, predicate="intersects")
    covered = np.asarray(shapely.area(shapely.intersection(bufs[e], outlines[b])),
                         dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.clip(np.where(area[b] > 0.0, covered / area[b], 0.0), 0.0, 1.0)
    cost = np.bincount(e, weights=share * w[b], minlength=len(segs)) / w.sum()
    weight = cost + length_tie * np.asarray(shapely.length(segs), dtype=np.float64)
    n = len(graph.pts)
    matrix = csr_matrix((np.concatenate([weight, weight]),
                         (np.concatenate([ui, uj]), np.concatenate([uj, ui]))), shape=(n, n))
    return _Graph(ui=ui, uj=uj, segs=segs, weight=weight, matrix=matrix, blocked=blocked)


def _targets(geo: gc.Geometry, r: _Reach, own: gc.Own, graph: RoutingGraph, g: _Graph,
             homes: NDArray[np.intp], reach: float, w: float) -> dict[int, _Targets]:
    """Per home, its targets: the ground that frees it is the sealed pocket of the complex its own
    ground lies in (the components, at the filtration width w, of its ring's triangles at baseline,
    `r`, that do not hold the street); a groundless home's (`own`) is within ring_m of its outline
    (its own footprint's included) and the sealed pocket its ring touches. A node or an edge within
    `reach` of that region is a target."""
    comp, street = gc.components(r.cx, w)
    ring = r.homes.ring.tocsr()
    nodes = shapely.points(graph.pts)
    out: dict[int, _Targets] = {}
    for h in homes:
        tris = ring.indices[ring.indptr[h]:ring.indptr[h + 1]]
        pockets = np.unique(comp[tris])
        pockets = pockets[pockets != street]
        sealed = np.flatnonzero(np.isin(comp, pockets))
        near = [shapely.buffer(geo.polys[h], r.cx.spec.ring_m)] if own.groundless[h] else []
        region = shapely.union_all([*near, *r.cx.tri_geom[sealed]])
        region = geo.utm(region)
        shapely.prepare(region)
        out[int(h)] = _Targets(
            nodes=np.flatnonzero(shapely.dwithin(region, nodes, reach) & ~g.blocked),
            edges=np.flatnonzero(shapely.dwithin(region, g.segs, reach)).astype(np.intp))
    return out


def _cheapest(graph: RoutingGraph, g: _Graph, t: _Targets, dist: NDArray[np.float64],
              pred: NDArray[np.int32], is_seed: NDArray[np.bool_], street: BaseGeometry, *,
              skip: set[int]) -> tuple[HeldRoad, int] | None:
    """The cheapest road to a target in this round's Dijkstra (`dist`, `pred`) and the node it
    ends on: to a target node, or through a target edge to its far end; street end first, with
    the stub to the street where it starts at a street node (as the methods' roads end), none
    ending on a node of `skip` (the home's targets whose road the metric's test refused). None if
    no target is reachable."""
    cand = [(float(dist[v]), int(v), -1) for v in t.nodes]
    for e in t.edges:
        a, b, we = int(g.ui[e]), int(g.uj[e]), float(g.weight[e])
        cand += [(float(dist[a]) + we, a, b), (float(dist[b]) + we, b, a)]
    for _cost, end, far in sorted(c for c in cand if np.isfinite(c[0])):
        if (far if far >= 0 else end) in skip:
            continue
        path = [end]
        while pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        path.reverse()                                      # the source (street or held) first
        if far >= 0:
            path.append(far)
        coords = [(float(graph.pts[k][0]), float(graph.pts[k][1])) for k in path]
        if is_seed[path[0]]:
            sp = nearest_points(Point(coords[0]), street)[1]
            coords.insert(0, (float(sp.x), float(sp.y)))
        coords = [c for i, c in enumerate(coords) if i == 0 or c != coords[i - 1]]
        if len(coords) >= 2:     # a street node exactly on the street, alone, makes no line
            return (HeldRoad(geometry=LineString(coords), nodes=tuple(int(k) for k in path)),
                    far if far >= 0 else end)
    return None


def lineup_free() -> FreeEnclosed:
    """The research lineup's free-first stage: FreeEnclosed under RUNTIME (metrics.RUNTIME), the
    area population routing and ordering, FREE_BUDGET, FREE_MARGIN_M and FREE_LENGTH_TIE."""
    import metrics
    return FreeEnclosed(budget=FREE_BUDGET, metric=metrics.RUNTIME,
                        population=common.AreaPopulation(), target_margin_m=FREE_MARGIN_M,
                        length_tie=FREE_LENGTH_TIE, measure=common.MEASURE)
