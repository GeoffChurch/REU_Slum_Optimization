"""ResistanceGreedyReblocker: one objective, optimized directly.

`clearance` + `LoopClosureRefiner` is two mechanisms bolted together — build a drainage tree
against access depth, then bolt on connectors chosen by bridges-removed-per-metre. Neither stage
optimizes the thing actually reported. This optimizes it directly: at every step add the road with
the best **permeability gain per metre**, and stop when that gain stops paying.

## Why permeability IS the resistance objective

`specs/2026-07-21-unified-resistance-objective-notes.md` proposed grounding the network at the
street and minimizing each parcel's effective resistance to ground, on the grounds that ONE
quantity captures both goals: poor access means electrically far from the street (external
connectivity), and a parcel on a spur has higher resistance than one on a loop (internal
connectivity / redundancy). It proposed writing a grounded-resistance scorer to do it.

That scorer shipped the next day under a different name. `permeability` solves `P = bᵀL⁻¹b` on the
parcel graph with the street eliminated as ground — the collective, contention-aware form of
exactly that quantity. The note predates it and calls for building what already exists, because
`commute_ratio` was the only resistance proxy at the time and has since been retired. So there is
no new solver to write; the objective is the shipped metric.

## Route (B), not (A)

The note gives two ways to optimize it. (A) is a convex program — total effective resistance is
convex in edge conductances (Ghosh-Boyd-Saberi 2008) — but "the ROUNDING is the hard part, because
a road is a combinatorial frontage PATH, not a free edge", and that rounding is unsolved. (B) is a
stochastic greedy: resistance-reduction is submodular, so sampling ~(N/k)·log(1/ε) candidates per
step gives (1−1/e−ε) at O(N log 1/ε) evaluations (Mirzasoleiman et al., arXiv:1409.7938) instead
of O(Nk). This is (B).

Candidate generation is deliberately the SAME as clearance's — one multi-source Dijkstra from the
current network per round, giving each parcel its cheapest connecting path — so the only thing
that differs from the shipped method is *which candidate gets chosen*. Depth-first versus
gain-per-metre, with everything else held equal.

Cost note: each candidate evaluation is one sparse solve, so this is `sample_size` solves per road.
Fine per block; a region-scale run needs the sample kept small or the incremental scorer the note
sketches.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.loop_closure import loop_candidates
from reblock.methods.substrates import ChordSubstrate, RoutingGraph, Substrate
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    PermeabilityParams,
    _footpath_conductance,
    _road_corridor,
    egress_power,
    lane_width,
    parcel_radii,
    permeability,
    road_conductance,
    with_width,
)


def _path_road(
    graph: RoutingGraph, pred: np.ndarray, start: int, rep: np.ndarray,
    street: shapely.geometry.base.BaseGeometry,
) -> LineString | None:
    """The traced Dijkstra path from a parcel back to the network, as a buildable LineString."""
    pathn = [start]
    while pred[pathn[-1]] >= 0:
        pathn.append(int(pred[pathn[-1]]))
    coords: list[tuple[float, float]] = [(float(rep[0]), float(rep[1]))]
    coords += [(float(graph.pts[k][0]), float(graph.pts[k][1])) for k in pathn]
    term = Point(graph.pts[pathn[-1]])
    if street.distance(term) <= graph.net_tol:
        sp = nearest_points(term, street)[1]
        coords.append((sp.x, sp.y))
    coords = [c for i, c in enumerate(coords) if i == 0 or c != coords[i - 1]]
    if len(coords) < 2:
        return None
    return LineString(coords)


def _mesh(block: Block, params: PermeabilityParams, adj: list[set[int]],
          radii: NDArray[np.float64],
          road_width_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, upgrade_gain, segment) for every adjacency edge: the conductance a two-way road of
    `road_width_m` would ADD to it, and the centroid-to-centroid segment a road must intersect.

    Mirrors `permeability.egress_power`'s mesh assembly exactly -- same adjacency, same
    `_footpath_conductance`, same `max(footpath, road)` switch -- because the scorer below is only
    valid if it differentiates the SAME Laplacian the metric solves. The gain is floored at zero
    for the same reason the metric takes a `max`: a road never makes an edge worse.
    """
    geoms = list(block.parcels.geometry)
    cent = [g.centroid for g in geoms]
    cx = np.array([c.x for c in cent]), np.array([c.y for c in cent])
    rows, cols, dists = [], [], []
    for i in range(len(geoms)):
        for j in adj[i]:
            if j <= i:
                continue
            d = float(np.hypot(cx[0][i] - cx[0][j], cx[1][i] - cx[1][j]))
            if d > 0.0:
                rows.append(i)
                cols.append(j)
                dists.append(d)
    ri = np.asarray(rows, dtype=np.int64)
    ci = np.asarray(cols, dtype=np.int64)
    di = np.asarray(dists, dtype=np.float64)
    if di.size == 0:
        return ri, ci, np.zeros(0), np.empty(0, dtype=object)
    road_g = road_conductance(params, np.full(di.size,
                                             lane_width(params, road_width_m)), di)
    foot_g = _footpath_conductance(di, radii[ri] + radii[ci], params.g_walk)
    segs = np.array([LineString([(cx[0][a], cx[1][a]), (cx[0][b], cx[1][b])])
                     for a, b in zip(ri.tolist(), ci.tolist(), strict=True)], dtype=object)
    return ri, ci, np.maximum(road_g - foot_g, 0.0), segs


def linearized_gain(
    v: np.ndarray, ri: np.ndarray, ci: np.ndarray, dg: np.ndarray,
    upgraded: np.ndarray,
) -> np.ndarray:
    """Per-edge first-order drop in dissipated power if that edge were upgraded to a road.

    P = b^T L^-1 b, and upgrading edge (i,j) by dg changes L by dg*(e_i - e_j)(e_i - e_j)^T. With
    v = L^-1 b already in hand, the first-order sensitivity is

        dP/d(dg) = -(v_i - v_j)^2      so      deltaP ~= -dg * (v_i - v_j)^2

    which costs ONE solve for v and then O(1) per edge -- no solve per candidate. The exact
    rank-1 value divides by (1 + dg * (e_i-e_j)^T L^-1 (e_i-e_j)) >= 1, so this OVERSTATES the
    gain and is a ranking heuristic, not a score. `ResistanceGreedyReblocker` uses it to shortlist
    and then re-scores the shortlist exactly, which is what keeps the selection honest.
    """
    gain = dg * (v[ri] - v[ci]) ** 2
    gain[upgraded] = 0.0                # already a road: upgrading again buys nothing
    return gain


@dataclass(frozen=True)
class ResistanceGreedyIdentity:
    substrate: Hashable
    max_roads: int
    sample_size: int
    min_gain_per_m: float
    seed: int
    loop_radius_m: float
    min_loop_len_m: float


@dataclass
class ResistanceGreedyReblocker:
    """Greedily add the road with the best permeability gain per metre."""

    substrate: Substrate = field(default_factory=ChordSubstrate)
    max_roads: int = 400
    shortlist: int = 6
    # OFF by default, and the reason is measured rather than assumed. Loop connectors ARE
    # generated in quantity when this is on (90-275 per round on 50-160 parcel blocks) and are
    # NEVER selected: an access road moves a parcel from footpath-only to road-adjacent, a large
    # first-order gain, while a connector only adds redundancy among already-served parcels, a
    # second-order one. Per metre, access dominates at every step until the gain floor stops the
    # greedy -- so enabling this doubled wall clock (9.6s vs 4.5s) and changed the output
    # bit-for-bit not at all. Kept because it is the honest test of "one objective over both move
    # types", and the answer it gives -- access first, redundancy never inside this budget -- is a
    # finding about the objective.
    loop_radius_m: float = 0.0
    min_loop_len_m: float = 40.0
    max_loop_candidates: int = 400
    min_gain_per_m: float = 1e-6
    seed: int = 0
    params: PermeabilityParams = field(default_factory=PermeabilityParams)
    # Total width of the roads this method emits; stamped on every one. The metric has no
    # global corridor to fall back on.
    road_width_m: float = DEFAULT_ROAD_WIDTH_M

    @property
    def identity(self) -> ResistanceGreedyIdentity | None:
        if self.substrate.identity is None:
            return None
        return ResistanceGreedyIdentity(
            substrate=self.substrate.identity, max_roads=int(self.max_roads),
            sample_size=int(self.shortlist), min_gain_per_m=float(self.min_gain_per_m),
            seed=int(self.seed), loop_radius_m=float(self.loop_radius_m),
            min_loop_len_m=float(self.min_loop_len_m))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        graph = self.substrate.build(block)
        crs = block.crs
        empty = with_width(gpd.GeoDataFrame(geometry=[], crs=crs), self.road_width_m)
        geoms = list(block.parcels.geometry)
        if len(geoms) == 0 or len(graph.pts) == 0:
            return self._proposal(block, empty, {"roads": 0, "stopped": "empty"})

        # Frozen once and reused for every candidate evaluation: the adjacency, the adaptive
        # corridor half-width and the no-roads baseline are properties of the BLOCK, not of the
        # road set, and recomputing them per candidate would dominate the cost.
        # STREET_TOL, matching `egress_power`'s own default -- NOT the road half-width. Building the
        # mesh at the road half-width (3.0 vs 0.5) gave a 6x looser adjacency than the evaluator
        # scores, so this method optimized a different Laplacian than the one it is graded
        # on -- exactly what `_mesh`'s docstring says must not happen.
        adj = parcel_adjacency(geoms, STREET_TOL)
        pradii = parcel_radii(block, self.params)

        street = unary_union(list(block.streets.geometry))
        net = np.flatnonzero(
            shapely.dwithin(shapely.points(graph.pts), street, graph.net_tol)).tolist()
        if not net:
            return self._proposal(block, empty, {"roads": 0, "stopped": "no street frontage"})

        pt_tree = cKDTree(graph.pts)
        reps = np.array([[g.representative_point().x, g.representative_point().y]
                         for g in geoms])
        starts = pt_tree.query(reps)[1]
        w = np.concatenate([graph.edist, graph.edist])
        csr = csr_matrix(
            (w, (np.concatenate([graph.rows, graph.cols]),
                 np.concatenate([graph.cols, graph.rows]))),
            shape=(len(graph.pts), len(graph.pts)))

        ri, ci, dg, segs = _mesh(block, self.params, adj, pradii, self.road_width_m)
        seg_tree = STRtree(list(segs)) if len(segs) else None

        roads: list[LineString] = []
        current = permeability(block, empty, self.params, adj=adj, radii=pradii)
        stopped = "max_roads"
        for _ in range(self.max_roads):
            _d, pred, _src = dijkstra(csr, indices=net, return_predecessors=True, min_only=True)
            pool = [i for i in range(len(geoms)) if pred[starts[i]] >= 0]
            if not pool:
                stopped = "no reachable parcel"
                break

            # ONE solve per round gives v = L^-1 b; every candidate is then scored in O(edges it
            # covers) by the first-order sensitivity, so ALL candidates are considered instead of
            # a random sample. See `linearized_gain`.
            built = with_width(gpd.GeoDataFrame(geometry=roads, crs=crs) if roads else empty,
                    self.road_width_m)
            _p, v = egress_power(block, built, self.params, adj=adj, radii=pradii)
            corridor = _road_corridor(built, self.road_width_m / 2.0)
            # One indexed query instead of a shapely call per mesh edge -- the same hot spot
            # `permeability._covered_edges` fixes, and this runs once per greedy round.
            upgraded = np.zeros(len(segs), dtype=bool)
            if corridor is not None and seg_tree is not None:
                upgraded[seg_tree.query(corridor, predicate="intersects")] = True
            edge_gain = linearized_gain(v, ri, ci, dg, upgraded)

            # Candidates are BOTH kinds of move, which is what makes this one objective rather
            # than two stages. Until loop connectors were added here the method was structurally a
            # tree builder -- every candidate attached an unserved parcel to the network, so it
            # could never choose redundancy however much the objective wanted it, and the note's
            # "one quantity captures access AND redundancy" was only half implemented.
            cands: list[LineString] = []
            for i in pool:
                road = _path_road(graph, pred, int(starts[i]), reps[i], street)
                if road is not None and road.length > 0:
                    cands.append(road)
            if roads and self.loop_radius_m > 0:
                built_gdf = with_width(gpd.GeoDataFrame(geometry=roads, crs=crs),
                        self.road_width_m)
                cands += [c for c, _u, _v in loop_candidates(
                    built_gdf, block, search_radius_m=self.loop_radius_m,
                    min_loop_len_m=self.min_loop_len_m, snap_lam=2.0,
                    max_candidates=self.max_loop_candidates)]

            ranked: list[tuple[float, LineString]] = []
            for road in cands:
                if road.length <= 0:
                    continue
                if seg_tree is None:
                    est = 0.0
                else:
                    hit = seg_tree.query(road.buffer(self.road_width_m / 2.0),
                                         predicate="intersects")
                    est = float(edge_gain[hit].sum()) / road.length
                ranked.append((est, road))
            if not ranked:
                stopped = "no candidate"
                break
            ranked.sort(key=lambda t: -t[0])

            # The linearization overstates gain (see `linearized_gain`), so it SHORTLISTS and the
            # exact metric decides -- `shortlist` exact solves per road instead of one per
            # candidate.
            best_gain, best_road, best_per_m = 0.0, None, 0.0
            for _est, road in ranked[:max(self.shortlist, 1)]:
                trial = with_width(gpd.GeoDataFrame(geometry=[*roads, road], crs=crs),
                    self.road_width_m)
                gain = permeability(block, trial, self.params, adj=adj, radii=pradii) - current
                per_m = gain / road.length
                if per_m > best_per_m:
                    best_gain, best_road, best_per_m = gain, road, per_m
            if best_road is None or best_per_m < self.min_gain_per_m:
                stopped = "gain below floor"
                break
            roads.append(best_road)
            current += best_gain
            # Extend the network so later roads branch off this one, exactly as clearance does.
            for x, y in best_road.coords:
                net.extend(pt_tree.query_ball_point([x, y], graph.net_tol))
            net = list(dict.fromkeys(net))

        gdf = with_width(gpd.GeoDataFrame(geometry=roads, crs=crs),
                  self.road_width_m)
        return self._proposal(block, gdf, {"roads": len(roads), "stopped": stopped,
                                           "permeability": float(current)})

    def _proposal(self, block: Block, roads: gpd.GeoDataFrame,
                  params: dict[str, object]) -> Proposal:
        pid = (f"resistance_greedy:{self.substrate.tag}:mr{self.max_roads}"
               f":s{self.shortlist}:g{self.min_gain_per_m:g}:seed{self.seed}")
        return Proposal(
            block_id=block.block_id, crs=block.crs, edges=None,
            roads=with_width(roads, self.road_width_m),
            proposal_id=pid, method="resistance_greedy",
            params={**params, "substrate": self.substrate.tag,
                    "shortlist": self.shortlist},
            block_identity=block.identity if self.identity is not None else None)
