"""A cycle-native reblocker: the CYCLE is the primitive, not the spur.

Every other method adds spurs -- a path from an unserved parcel back to the network -- which is why
they all score at the tree end of the directed penalty (5.47, against a pure cycle's 2.27). A
method whose atomic move is a LOOP is bridgeless by construction, needs no repair pass, and is
strongly orientable, so every road it emits can be made one-way. That last property was the point
of the exercise and it did NOT pay -- see `notes/2026-07-31-one-way-is-dominated.md`; the value that
survives is the bridgelessness itself.

## The move

Each step adds one cycle: a Dijkstra path from the street out to a chosen parcel, plus a SECOND
path from that parcel back to the street that avoids the first one's edges. The union is a cycle
through the street. Candidates are scored by permeability gain per unit displacement -- the same
currency the lenses use -- and the best is committed.

Avoiding the outbound edges is what makes the return path a genuine alternate route rather than a
retrace; without it the "cycle" is a there-and-back spur and the method degenerates to clearance.

## Measured (22 blocks, matched displacement 10%)

    method             penalty  bridges    perm   dispB   road_m
    clearance_looped    5.4762   0.1766  0.6752  0.1049       78
    greedy_arterial     5.4754   0.1952  0.6218  0.1288      136
    resistance_lp       5.4716   0.3536  0.7982  0.0431      242
    cycle_native        5.0323   0.0000  0.7464  0.0625      193

Beats the flagship on BOTH lenses -- permeability at matched displacement 13/22 (+0.0198), and
displacement to reach P* 16/20 (-0.0442) -- and is the only method with ZERO bridges, so it is the
only one every road of which can be made one-way with no repair pass.

`resistance_lp` still leads both reported lenses. This method's distinct value is the circulation
axis: partial bridge reduction buys almost nothing (`resistance_lp` has the HIGHEST bridge fraction
at 0.354 and still sits with the pack at 5.47), while reaching exactly zero breaks out.

Its cost is METRES -- 193 against the flagship's 78 at matched displacement. An earlier two-phase
probe suggested circulation costs ~16% of permeability; that was an artifact of bolting connectors
onto a finished tree. Choosing cycles from the start finds loops that also serve parcels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union

from reblock.budget import building_radii, displacement
from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.substrates import ChordSubstrate, RoutingGraph, Substrate
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    PermeabilityParams,
    _adaptive_r0,
    permeability,
    with_width,
)


def _trace(graph: RoutingGraph, pred: np.ndarray, start: int) -> list[int]:
    path = [start]
    while pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return path


def _geom(graph: RoutingGraph, nodes: list[int],
          street: shapely.geometry.base.BaseGeometry) -> LineString | None:
    cs = [(float(graph.pts[k][0]), float(graph.pts[k][1])) for k in nodes]
    term = Point(graph.pts[nodes[-1]])
    if street.distance(term) <= graph.net_tol:
        sp = nearest_points(term, street)[1]
        cs.append((sp.x, sp.y))
    cs = [c for i, c in enumerate(cs) if i == 0 or c != cs[i - 1]]
    return LineString(cs) if len(cs) >= 2 else None


@dataclass
class CycleNativeReblocker:
    """Greedy over CYCLES: each move is a loop from the street back to the street."""

    substrate: Substrate = field(default_factory=ChordSubstrate)
    max_displacement: float = 0.10
    shortlist: int = 8
    params: PermeabilityParams = field(default_factory=PermeabilityParams)
    # Total width of the roads this method emits; stamped on every one. The metric has no
    # global corridor to fall back on.
    road_width_m: float = DEFAULT_ROAD_WIDTH_M
    identity = None

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        crs = block.crs
        empty = gpd.GeoDataFrame(geometry=[], crs=crs)
        graph = self.substrate.build(block)
        geoms = list(block.parcels.geometry)
        if not geoms or len(graph.pts) == 0:
            return self._out(block, empty)

        adj = parcel_adjacency(geoms, STREET_TOL)
        r0 = _adaptive_r0(block, self.params)
        radii = building_radii(block.building_points)
        n_b = max(len(block.building_points), 1)
        street = unary_union(list(block.streets.geometry))
        seeds = np.flatnonzero(
            shapely.dwithin(shapely.points(graph.pts), street, graph.net_tol)).tolist()
        if not seeds:
            return self._out(block, empty)

        pt_tree = cKDTree(graph.pts)
        reps = np.array([[g.representative_point().x, g.representative_point().y] for g in geoms])
        starts = pt_tree.query(reps)[1]
        n_pts = len(graph.pts)
        w = np.concatenate([graph.edist, graph.edist])
        rr = np.concatenate([graph.rows, graph.cols])
        cc = np.concatenate([graph.cols, graph.rows])

        roads: list[LineString] = []
        cur_p = float(permeability(block, empty, self.params, adj=adj, r0=r0))
        for _ in range(60):
            spent = displacement(block.building_points, radii,
                                 with_width(gpd.GeoDataFrame(geometry=roads, crs=crs),
                                            self.road_width_m)) / n_b \
                if roads else 0.0
            if spent >= self.max_displacement:
                break
            net = list(seeds)
            for road in roads:
                for x, y in road.coords:
                    net.extend(pt_tree.query_ball_point([x, y], graph.net_tol))
            net = list(dict.fromkeys(net))
            csr = csr_matrix((w, (rr, cc)), shape=(n_pts, n_pts))
            _d, pred, _s = dijkstra(csr, indices=net, return_predecessors=True, min_only=True)
            pool = [i for i in range(len(geoms)) if pred[starts[i]] >= 0]
            if not pool:
                break

            # deepest-first shortlist, then score each candidate CYCLE exactly
            depth = np.array([_d[starts[i]] if np.isfinite(_d[starts[i]]) else -1.0 for i in pool])
            order = [pool[k] for k in np.argsort(-depth)[: max(self.shortlist, 1)]]
            best, best_val = None, 0.0
            for i in order:
                out_nodes = _trace(graph, pred, int(starts[i]))
                out_geom = _geom(graph, out_nodes, street)
                if out_geom is None:
                    continue
                # RETURN leg: same Dijkstra, but the outbound edges are removed, so the way back is
                # a genuine alternate route rather than a retrace.
                used = {(min(a, b), max(a, b))
                        for a, b in zip(out_nodes, out_nodes[1:], strict=False)}
                keep = np.array([(min(a, b), max(a, b)) not in used
                                 for a, b in zip(rr.tolist(), cc.tolist(), strict=True)])
                csr2 = csr_matrix((w[keep], (rr[keep], cc[keep])), shape=(n_pts, n_pts))
                _d2, pred2, _s2 = dijkstra(csr2, indices=net, return_predecessors=True,
                                           min_only=True)
                back = None
                if pred2[int(starts[i])] >= 0:
                    back = _geom(graph, _trace(graph, pred2, int(starts[i])), street)
                cand = [out_geom] + ([back] if back is not None else [])
                trial = with_width(gpd.GeoDataFrame(geometry=[*roads, *cand], crs=crs),
                                   self.road_width_m)
                d = displacement(block.building_points, radii, trial) / n_b
                if d > self.max_displacement:
                    continue
                gain = float(permeability(block, trial, self.params, adj=adj, r0=r0)) - cur_p
                cost = max(d - spent, 1e-9)
                if gain > 0 and gain / cost > best_val:
                    best, best_val = (cand, gain + cur_p), gain / cost
            if best is None:
                break
            roads.extend(best[0])
            cur_p = best[1]
        return self._out(block, gpd.GeoDataFrame(geometry=roads, crs=crs))

    def _out(self, block: Block, roads: gpd.GeoDataFrame) -> Proposal:
        out = with_width(roads, self.road_width_m)
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=out, edges=None,
            proposal_id=f"cycle_native:d{self.max_displacement:g}", method="cycle_native",
            params={"roads": len(out), "road_width_m": self.road_width_m}, block_identity=None)
