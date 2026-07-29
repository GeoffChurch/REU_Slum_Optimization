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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.substrates import ChordSubstrate, RoutingGraph, Substrate
from reblock.permeability import PermeabilityParams, _adaptive_r0, permeability


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


@dataclass(frozen=True)
class ResistanceGreedyIdentity:
    substrate: Hashable
    max_roads: int
    sample_size: int
    min_gain_per_m: float
    seed: int


@dataclass
class ResistanceGreedyReblocker:
    """Greedily add the road with the best permeability gain per metre."""

    substrate: Substrate = field(default_factory=ChordSubstrate)
    max_roads: int = 400
    sample_size: int = 24
    min_gain_per_m: float = 1e-6
    seed: int = 0
    params: PermeabilityParams = field(default_factory=PermeabilityParams)

    @property
    def identity(self) -> ResistanceGreedyIdentity | None:
        if self.substrate.identity is None:
            return None
        return ResistanceGreedyIdentity(
            substrate=self.substrate.identity, max_roads=int(self.max_roads),
            sample_size=int(self.sample_size), min_gain_per_m=float(self.min_gain_per_m),
            seed=int(self.seed))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        graph = self.substrate.build(block)
        crs = block.crs
        empty = gpd.GeoDataFrame(geometry=[], crs=crs)
        geoms = list(block.parcels.geometry)
        if len(geoms) == 0 or len(graph.pts) == 0:
            return self._proposal(block, empty, {"roads": 0, "stopped": "empty"})

        # Frozen once and reused for every candidate evaluation: the adjacency, the adaptive
        # corridor half-width and the no-roads baseline are properties of the BLOCK, not of the
        # road set, and recomputing them per candidate would dominate the cost.
        adj = parcel_adjacency(geoms, self.params.corridor_m)
        r0 = _adaptive_r0(block, self.params)

        street = unary_union(list(block.streets.geometry))
        net = np.flatnonzero(
            shapely.dwithin(shapely.points(graph.pts), street, graph.net_tol)).tolist()
        if not net:
            return self._proposal(block, empty, {"roads": 0, "stopped": "no street frontage"})

        pt_tree = cKDTree(graph.pts)
        reps = np.array([[g.representative_point().x, g.representative_point().y]
                         for g in geoms])
        starts = pt_tree.query(reps)[1]
        rng = np.random.default_rng(self.seed)
        w = np.concatenate([graph.edist, graph.edist])
        csr = csr_matrix(
            (w, (np.concatenate([graph.rows, graph.cols]),
                 np.concatenate([graph.cols, graph.rows]))),
            shape=(len(graph.pts), len(graph.pts)))

        roads: list[LineString] = []
        current = permeability(block, empty, self.params, adj=adj, r0=r0)
        stopped = "max_roads"
        for _ in range(self.max_roads):
            _d, pred, _src = dijkstra(csr, indices=net, return_predecessors=True, min_only=True)
            pool = [i for i in range(len(geoms)) if pred[starts[i]] >= 0]
            if not pool:
                stopped = "no reachable parcel"
                break
            # Stochastic greedy: a random sample per step is what buys (1-1/e-eps) at
            # O(N log 1/eps) evaluations instead of O(Nk) -- see the module docstring.
            k = min(self.sample_size, len(pool))
            best_gain, best_road, best_per_m = 0.0, None, 0.0
            for i in rng.choice(pool, size=k, replace=False):
                road = _path_road(graph, pred, int(starts[i]), reps[i], street)
                if road is None or road.length <= 0:
                    continue
                trial = gpd.GeoDataFrame(geometry=[*roads, road], crs=crs)
                gain = permeability(block, trial, self.params, adj=adj, r0=r0) - current
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

        gdf = gpd.GeoDataFrame(geometry=roads, crs=crs)
        return self._proposal(block, gdf, {"roads": len(roads), "stopped": stopped,
                                           "permeability": float(current)})

    def _proposal(self, block: Block, roads: gpd.GeoDataFrame,
                  params: dict[str, object]) -> Proposal:
        pid = (f"resistance_greedy:{self.substrate.tag}:mr{self.max_roads}"
               f":s{self.sample_size}:g{self.min_gain_per_m:g}:seed{self.seed}")
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="resistance_greedy",
            params={**params, "substrate": self.substrate.tag,
                    "sample_size": self.sample_size},
            block_identity=block.identity if self.identity is not None else None)
