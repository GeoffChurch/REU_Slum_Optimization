"""GreedyArterialReblocker: greedily insert the single straight arterial with the best
objective gain per meter, one at a time, until a road budget runs out. Two modes -- buildable
(snapped to the parcel-boundary graph; the shippable navigability method) and aspirational (ideal
chords; a diagnostic isolating the effect of frontage-snapping, NOT a universal directness ceiling
-- see the design doc's correction note). Candidates are through-roads (network<->network) + spurs
(network->deep pocket); continuations are through-roads from committed-segment endpoints (always
anchors), so a spur completes into a through-road for free and crossings planarize into true
intersections. See docs/superpowers/specs/2026-07-09-greedy-arterial-reblocker-design.md.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.budget import (
    _BlockScoringContext,
    access_burden,
    displacement_count,
    road_drainage,
)
from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.dijkstra import _boundary_graph, _rnd


def _xy(c: tuple[float, ...]) -> tuple[float, float]:
    """First two components of a coordinate tuple (shapely yields 3-tuples for Z-aware
    geometry; every geometry here is 2-D, so drop anything past x, y)."""
    return (c[0], c[1])


def _anchor_points(network: Sequence[BaseGeometry], n: int) -> list[tuple[float, float]]:
    """`n` points sampled evenly by arc-length along the merged network, plus every network
    vertex (so committed-segment endpoints are always anchors -> continuations come for free).
    `unary_union` explodes any Multi* input, so streets given as a MultiLineString (a block with
    a hole/courtyard) are handled. _rnd-snapped, de-duplicated, sorted for determinism."""
    merged = unary_union(network)
    lines = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    pts: set[tuple[float, float]] = set()
    for ln in lines:
        pts.update(_rnd(_xy(c)) for c in ln.coords)                  # vertices
    total = sum(ln.length for ln in lines)
    if total > 0 and n > 0:
        step = total / n
        for ln in lines:
            d = 0.0
            while d <= ln.length:
                pts.add(_rnd(_xy(ln.interpolate(d).coords[0])))
                d += step
    return sorted(pts)


def _deep_targets(block: Block, roads: GeoDataFrame | None, k: int,
                   adj: list[set[int]]) -> list[tuple[float, float]]:
    """Representative points of the k deepest-access parcels (spur targets), _rnd-snapped."""
    depths = parcel_access_layers(block, roads, tol=STREET_TOL, adj=adj)
    order = depths.sort_values(ascending=False, kind="stable")
    id_to_pos = {pid: i for i, pid in enumerate(block.parcels["parcel_id"])}
    geoms = list(block.parcels.geometry)
    out: list[tuple[float, float]] = []
    for pid in list(order.index)[:k]:
        rep = geoms[id_to_pos[pid]].representative_point()
        out.append(_rnd(_xy(rep.coords[0])))
    return out


def _candidate_chords(anchors: list[tuple[float, float]],
                       targets: list[tuple[float, float]]) -> list[LineString]:
    """Through-roads (anchor pairs) + spurs (anchor -> deep target). De-duplicated, sorted."""
    seen: set[frozenset[tuple[float, float]]] = set()
    chords: list[LineString] = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            key = frozenset((a, b))
            if a != b and key not in seen:
                seen.add(key)
                pair: list[tuple[float, float]] = sorted((a, b))
                chords.append(LineString(pair))
        for t in targets:
            key = frozenset((a, t))
            if a != t and key not in seen:
                seen.add(key)
                spur: list[tuple[float, float]] = sorted((a, t))
                chords.append(LineString(spur))
    return sorted(chords, key=lambda ls: ls.wkt)


@dataclass
class _SnapGraph:
    """`_boundary_graph(block.parcels)` plus everything `_snap` needs precomputed ONCE per block:
    the nearest-node lookup tree and, for every graph edge, its midpoint (as a shapely `Point`,
    for the vectorized ufunc below) and length. Building this once per block -- instead of per
    candidate chord -- lifts the per-edge Python-loop `mid.distance(chord)` (the profiled `_snap`
    hot cost) out of the greedy's ~thousands-of-candidates inner loop."""
    g: nx.Graph
    node_tree: STRtree
    nodes: list[tuple[float, float]]
    edges: list[tuple[tuple[float, float], tuple[float, float]]]
    mid_points: np.ndarray               # shapely Point per edge, aligned with `edges`
    lengths: np.ndarray                  # edge length (== g[u][v]["weight"]), aligned with `edges`


def _snap_graph(g: nx.Graph) -> _SnapGraph:
    nodes = list(g.nodes)
    node_tree = STRtree([Point(nd) for nd in nodes])
    edges = list(g.edges())
    xs = np.array([(u[0] + v[0]) / 2 for u, v in edges], dtype=float)
    ys = np.array([(u[1] + v[1]) / 2 for u, v in edges], dtype=float)
    lengths = np.array([g[u][v]["weight"] for u, v in edges], dtype=float)
    return _SnapGraph(g, node_tree, nodes, edges, shapely.points(xs, ys), lengths)


def _snap(chord: LineString, sg: _SnapGraph, lam: float) -> LineString | None:
    """Buildable realization: the boundary-graph path between the chord endpoints' nearest
    nodes that hugs the ideal line (edge cost = length + lam * dist(edge midpoint, chord)).
    None if the endpoints snap to the same node or no path exists.

    The per-edge weight is computed ONCE per chord via shapely's vectorized `shapely.distance`
    ufunc over the block's precomputed `edge_midpoints` (bit-identical to the scalar
    `Point.distance(chord)` the naive per-edge loop would call -- both call into the same GEOS
    routine) instead of a Python-loop shapely call per edge per Dijkstra relaxation. The result is
    folded into an `{(u, v): weight}` dict (both directions, since the graph is undirected and
    `nx.shortest_path`'s callback may see either); the weight callback only looks it up, so the
    same `nx.shortest_path` call, weights, and (hence) equal-cost tie-break as before -- identical
    path geometry, minus the per-edge shapely cost."""
    p, q = _rnd(_xy(chord.coords[0])), _rnd(_xy(chord.coords[-1]))
    np_ = sg.nodes[int(sg.node_tree.nearest(Point(p)))]
    nq_ = sg.nodes[int(sg.node_tree.nearest(Point(q)))]
    if np_ == nq_:
        return None

    dists = shapely.distance(sg.mid_points, chord)
    weights = sg.lengths + lam * dists
    weight_map: dict[tuple[tuple[float, float], tuple[float, float]], float] = {}
    for (u, v), wt in zip(sg.edges, weights, strict=True):
        fwt = float(wt)
        weight_map[(u, v)] = fwt
        weight_map[(v, u)] = fwt

    def w(u: tuple[float, float], v: tuple[float, float], d: dict[str, float]) -> float:
        del d
        return weight_map[(u, v)]

    try:
        path = nx.shortest_path(sg.g, np_, nq_, weight=w)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return LineString([tuple(node) for node in path])


def _merge(lines: list[LineString]) -> BaseGeometry | None:
    """`unary_union(lines)`, or `None` for an empty list (the incremental-`_planarize` base)."""
    return unary_union(lines) if lines else None


def _explode(merged: BaseGeometry | None, crs: CRS) -> GeoDataFrame:
    """Explode a (possibly `None`) merged/noded geometry into a one-row-per-LineString
    GeoDataFrame."""
    if merged is None:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    parts: list[BaseGeometry] = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    rows = [ln for ln in parts if "LineString" in ln.geom_type and ln.length > 0]
    return gpd.GeoDataFrame({"geometry": rows}, geometry="geometry", crs=crs)


def _planarize(lines: list[LineString], crs: CRS) -> GeoDataFrame:
    """unary_union the lines (nodes crossings), explode to LineStrings, one row each."""
    return _explode(_merge(lines), crs)


def _union_with(base_merged: BaseGeometry | None, real: LineString) -> BaseGeometry:
    """Incremental planarize: node `real` against the already-merged `base_merged` (or just
    `real` alone if there is no base yet) instead of re-unioning the whole committed list. Matches
    `_StepContext.score_candidate`'s road-graph merge exactly (`budget.py`) -- bit-exact for
    BUILDABLE trials, which meet the committed/street network only at shared boundary-graph
    vertices, so this incremental two-stage union nodes identically to a one-shot union of the
    full list. It is NOT bit-exact for aspirational free chords crossing a committed edge at a
    float interior point ("Bug 2" -- see `_StepContext`'s docstring), so callers must gate this
    on `mode == "buildable"` and use the full `_planarize(committed + [real], ...)` for
    aspirational."""
    return unary_union([base_merged, real]) if base_merged is not None else unary_union([real])


def _score(objective: str, block: Block, roads: GeoDataFrame, adj: list[set[int]],
           base_burden: float, ctx: _BlockScoringContext | None) -> float:
    """Objective value of the road set (higher = better). Mirrors the budget metrics with a
    cached parcel-adjacency for the greedy's inner loop. For efficiency/directness, scores through
    the per-block `ctx` (frozen constants, one context per block, full entry re-derivation per
    candidate) -- equivalent to `network_efficiency(block, roads)`."""
    if objective == "access":
        if base_burden == 0.0:
            return 0.0
        depths = parcel_access_layers(block, roads, tol=STREET_TOL, adj=adj,
                                      unreached_depth=len(block.parcels) + 1)
        return 1.0 - access_burden(depths) / base_burden
    assert ctx is not None
    e, direct = ctx.score(roads)
    return e if objective == "efficiency" else direct


def _greedy_arterials(block: Block, *, mode: str, objective: str, n_anchors: int = 32,
                      top_k: int = 8, lam: float = 2.0, max_roads: int = 15,
                      cost: str = "length", corridor_m: float = 3.0) -> GeoDataFrame:
    """Greedily commit the straight arterial with the best objective gain per unit cost until
    `max_roads` are placed or no candidate improves. `mode` in {"buildable", "aspirational"}.
    `cost` in {"length" (Delta-benefit/metre), "displacement" (Delta-benefit/building newly
    inside `corridor_m` of any committed road, via block.building_points -- see
    budget.displacement_count)}. A beneficial candidate whose denominator is zero (a
    zero-length road can't occur -- filtered below -- but a zero-marginal-displacement road
    can) ranks ABOVE every positive-denominator candidate (infinite gain) rather than being
    divided by zero or skipped: take the free navigability first."""
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=len(block.parcels) + 1))
    g = _boundary_graph(block.parcels)
    sg = _snap_graph(g)                    # precomputed once per block -- see `_snap`
    # Raw street geometries (may be a MultiLineString for a holed/courtyard block) -- do NOT
    # filter to LineString, or Multi* streets get dropped and the proposal comes back empty;
    # _anchor_points explodes Multi* internally.
    streets: list[BaseGeometry] = list(block.streets.geometry)
    # ONE scoring context per block (frozen reps/sources/src_euclid/street geometry), shared by
    # every candidate score below -- only built for the metric objectives that use it.
    ctx = (_BlockScoringContext(block) if objective in ("efficiency", "directness") else None)

    committed: list[LineString] = []                        # realized geometry, in commit order
    while len(committed) < max_roads:
        network: list[BaseGeometry] = [*streets, *committed]
        anchors = _anchor_points(network, n_anchors)
        base_merged = _merge(committed)              # unary_union(committed), once per step
        base = _explode(base_merged, block.crs)
        base_val = _score(objective, block, base, adj, base_burden, ctx)
        curr_roads = base if len(committed) else None
        targets = _deep_targets(block, curr_roads, top_k, adj)
        committed_disp = (displacement_count(block.building_points, base, corridor_m)
                          if cost == "displacement" else 0)
        # Route per-candidate scoring by mode. BUILDABLE trials are boundary-snapped (they join the
        # committed/street network at shared graph vertices), so the incremental
        # `step.score_candidate` is bit-exact to `_score(objective, _planarize(committed+[real]))`
        # while skipping the per-candidate full entry re-derivation -- the perf win. Likewise, a
        # buildable trial's noding is bit-exact under `_union_with`'s incremental `unary_union`
        # (§4 -- meets the network only at shared vertices), so the `access`-objective buildable
        # path (no `ctx`, so no `step`) uses that incremental union too. ASPIRATIONAL trials are
        # free chords crossing committed edges at float interior points, where the incremental
        # planarize noding is NOT bit-exact (design "Bug 2"), so those always use the full
        # `_planarize(committed + [real])` re-union below (still frozen-constants fast via `ctx`
        # when scored, for efficiency/directness).
        step = ctx.step(base) if (ctx is not None and mode == "buildable") else None

        best_gain, best_real = 0.0, None
        for chord in _candidate_chords(anchors, targets):
            real = chord if mode == "aspirational" else _snap(chord, sg, lam)
            if real is None or real.length == 0:
                continue
            trial: GeoDataFrame | None = None
            if step is not None:
                e, direct = step.score_candidate(real)
                raw = (e if objective == "efficiency" else direct) - base_val
            elif mode == "buildable":
                trial = _explode(_union_with(base_merged, real), block.crs)
                raw = _score(objective, block, trial, adj, base_burden, ctx) - base_val
            else:
                trial = _planarize(committed + [real], block.crs)
                raw = _score(objective, block, trial, adj, base_burden, ctx) - base_val
            if cost == "displacement":
                if trial is None:
                    trial = _explode(_union_with(base_merged, real), block.crs)  # step -> buildable
                denom = float(displacement_count(block.building_points, trial, corridor_m)
                             - committed_disp)
            else:
                denom = real.length
            gain = float("inf") if (denom <= 0 and raw > 0) else (raw / denom if denom > 0 else 0.0)
            if gain > best_gain or (best_real is not None and gain == best_gain
                                    and real.wkt < best_real.wkt):
                best_gain, best_real = gain, real
        if best_real is None:                               # no candidate improves -> stop
            break
        committed.append(best_real)

    roads = _planarize(committed, block.crs)
    roads["drain"] = road_drainage(block, roads) if len(roads) else []
    return roads


@dataclass
class GreedyArterialReblocker:
    mode: str = "buildable"          # "buildable" | "aspirational"
    objective: str = "directness"    # "access" | "efficiency" | "directness"
    n_anchors: int = 32
    top_k: int = 8
    lam: float = 2.0
    max_roads: int = 15
    # "length" (Delta-benefit/metre) | "displacement" (Delta-benefit/building, see budget.py)
    cost: str = "length"
    corridor_m: float = 3.0   # road half-width + setback; the displacement corridor

    @property
    def identity(self) -> tuple[str, str, str, str, float]:
        # corridor_m changes which roads win only under cost="displacement"; hold it fixed
        # otherwise so length-cost methods stay corridor-independent in the derive cache (two
        # methods differing only in corridor_m must NOT share a cached proposal when it matters).
        corridor_key = self.corridor_m if self.cost == "displacement" else 0.0
        return ("greedy_arterial", self.mode, self.objective, self.cost, corridor_key)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        roads = _greedy_arterials(block, mode=self.mode, objective=self.objective,
                                  n_anchors=self.n_anchors, top_k=self.top_k, lam=self.lam,
                                  max_roads=self.max_roads, cost=self.cost,
                                  corridor_m=self.corridor_m)
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=f"greedy_arterial_{self.mode}_{self.objective}", method="greedy_arterial",
            params={"segments": len(roads), "mode": self.mode, "objective": self.objective,
                    "cost": self.cost, "corridor_m": self.corridor_m},
            block_identity=block.identity)
