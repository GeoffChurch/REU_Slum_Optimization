"""Route (A): choose the whole road set at once with an LP, budgeting the currency actually scored.

The greedy's gap to clearance is myopia, not search -- shortlist 20 scored WORSE than shortlist 6.
So the fix has to be a formulation that sees the whole set. The design note's obstacle was rounding:
"a road is a combinatorial frontage PATH, not a free edge", so an LP over free edge conductances
gives a fractional answer with no buildable meaning.

The way around it is to never make edge conductances the decision variables. Decide over PATHS.

    x_p in [0,1]   candidate road p -- a traced Dijkstra path from a parcel back to the network
    z_s in [0,1]   substrate segment s is built
    y_e in [0,1]   parcel-adjacency edge e is upgraded from footpath to road
    u_b in [0,1]   building b is displaced

    max  sum_e g_e y_e                        g_e = dg_e (v_i - v_j)^2, the linearized gain
    s.t. sum_b u_b   <= D * n_buildings       THE LENS'S OWN BUDGET
         sum_s len_s z_s <= B_m               buildability cap, so gap roads are not free
         x_p <= z_s           for s in p      choosing a path builds every segment of it
         z_s <= sum_{p ni s} x_p              a segment exists only because some path wants it
         y_e <= sum_{s ni e} z_s              an adjacency edge upgrades only if a road covers it
         c_b(s) z_s <= u_b    for s near b    displacement is a MAX over segments, not a sum

Four things that matter, none of which the greedy can express:

1. **The budget is displacement, not metres.** Both lenses score displacement; metres are only a
   proxy, and a bad one -- the same metres cost wildly different displacement depending on whether
   they run down a gap or through the dense interior. A metre-budgeted version of this LP sat at
   D = 0.080-0.118 where clearance sat at 0.060-0.096, and lost on the lens for that reason alone.
2. **Displacement's union discount is modelled exactly.** `budget.displacement` measures each
   building against the UNION corridor, so its cost is `max_s c_b(s)`, not `sum_s`. The `u_b`
   constraint family is that max. Two roads flanking one building pay for it once. A greedy cannot
   do this -- marginal displacement is not CELF-safe, which is exactly why `greedy_arterial` had to
   abandon it for `repulsion`. Here it is a linear constraint and costs nothing.
3. **Sharing is credited on both sides.** Parcels whose paths share a trunk pay for it once
   (`z_s` appears once in the budget) and count its gain once (`y_e <= 1`).
4. **Connectivity is structural, not a rounding repair.** `x_p <= z_s` means any solution with
   `x_p > 0` has bought the whole path back to the street, and `z_s <= sum x_p` closes the
   converse loophole -- without it the LP sets `z_s = 1` on high-gain segments with every `x_p = 0`,
   buying gain without buying a road, which is the fractional nonsense route (A) was shelved for.

The objective is a linearization, so `chunks` spends the budget in K instalments, re-linearizing
between them: K=1 is the pure one-shot LP, large K approaches the greedy.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import shapely
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union

from reblock.budget import building_radii
from reblock.contracts import Block, Proposal
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.resistance_greedy import _mesh, linearized_gain
from reblock.methods.substrates import ChordSubstrate, RoutingGraph, Substrate
from reblock.permeability import (
    PermeabilityParams,
    _adaptive_r0,
    _road_corridor,
    egress_power,
    permeability,
)

CORRIDOR_M = 3.0
SegId = tuple[str, int, int]


def _path_segments(
    graph: RoutingGraph, pred: np.ndarray, start: int, parcel: int, rep: np.ndarray,
    street: shapely.geometry.base.BaseGeometry,
) -> list[tuple[SegId, LineString]]:
    """A parcel's path back to the network, as identified segments ordered STREET-FIRST.

    Identity is what makes sharing work: two parcels traced through the same substrate edge return
    the same `("e", u, v)` id, so the LP pays for it once. The two stubs cannot share -- the inlet
    is unique to the parcel, the outlet to its terminal node -- and are keyed accordingly.

    Street-first order means every PREFIX of the list is a connected road reaching the street,
    which is what lets the rounding commit part of a path.
    """
    pathn = [start]
    while pred[pathn[-1]] >= 0:
        pathn.append(int(pred[pathn[-1]]))
    out: list[tuple[SegId, LineString]] = []
    term = Point(graph.pts[pathn[-1]])
    if street.distance(term) <= graph.net_tol:
        sp = nearest_points(term, street)[1]
        if sp.distance(term) > 0:
            out.append((("out", pathn[-1], -1), LineString([(sp.x, sp.y), (term.x, term.y)])))
    for a, b in zip(pathn[-1:0:-1], pathn[-2::-1], strict=True):
        pa, pb = graph.pts[a], graph.pts[b]
        if pa[0] != pb[0] or pa[1] != pb[1]:
            out.append((("e", min(a, b), max(a, b)),
                        LineString([(pa[0], pa[1]), (pb[0], pb[1])])))
    tail = Point(graph.pts[start])
    if tail.distance(Point(rep[0], rep[1])) > 0:
        out.append((("in", parcel, -1),
                    LineString([(tail.x, tail.y), (float(rep[0]), float(rep[1]))])))
    return out


def segment_displacement(
    seg_geom: list[LineString], pts: gpd.GeoDataFrame, radii: np.ndarray, corridor_m: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per segment, `(building indices, c_b)` for every building its corridor alone would graze.

    Mirrors `budget.displacement` term for term -- `c = clip(1 - d/r)` with `d` measured to the
    corridor, `r == 0` counting iff `d <= 0` -- so that the max of these over the chosen segments
    is the exact union displacement, not an approximation of it.
    """
    if len(pts) == 0:
        return [(np.zeros(0, dtype=np.int64), np.zeros(0)) for _ in seg_geom]
    xy = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
    tree = cKDTree(xy)
    rmax = float(radii.max()) if radii.size else 0.0
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for g in seg_geom:
        near = np.asarray(tree.query_ball_point(np.asarray(g.coords), corridor_m + rmax,
                                                return_sorted=False), dtype=object)
        idx = np.unique(np.concatenate([np.asarray(a, dtype=np.int64) for a in near])
                        ) if near.size else np.zeros(0, dtype=np.int64)
        if idx.size == 0:
            out.append((idx, np.zeros(0)))
            continue
        d = np.maximum(shapely.distance(shapely.points(xy[idx]), g) - corridor_m, 0.0)
        r = radii[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(r > 0.0, 1.0 - d / r, np.where(d <= 0.0, 1.0, 0.0))
        c = np.clip(c, 0.0, 1.0)
        keep = c > 0.0
        out.append((idx[keep], c[keep]))
    return out


def solve_coverage_lp(
    path_segs: list[list[int]], seg_len: np.ndarray, seg_edges: list[np.ndarray],
    seg_disp: list[tuple[np.ndarray, np.ndarray]], edge_gain: np.ndarray,
    n_buildings: int, base_c: np.ndarray, disp_budget: float, len_budget: float,
    length_price: float = 0.0,
) -> np.ndarray:
    """The LP above. Returns `z`, the fractional build level of every segment.

    Variable layout is [x (paths) | z (segments) | y (adjacency edges) | u (buildings)], and every
    constraint is a row of A_ub in that block order. `base_c` is the displacement already incurred
    by roads built in earlier chunks; it becomes the lower bound on `u`, so the LP neither
    re-charges for it nor gets to pretend it is free.
    """
    n_p, n_s, n_e = len(path_segs), len(seg_len), len(edge_gain)
    n_b = n_buildings
    zoff, yoff, uoff = n_p, n_p + n_s, n_p + n_s + n_e
    n_var = uoff + n_b
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    ub: list[float] = []

    def row(entries: list[tuple[int, float]], rhs: float) -> None:
        r = len(ub)
        for c, v in entries:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        ub.append(rhs)

    if n_b:
        row([(uoff + b, 1.0) for b in range(n_b)], disp_budget)
    row([(zoff + s, float(seg_len[s])) for s in range(n_s)], len_budget)
    holders: list[list[int]] = [[] for _ in range(n_s)]
    for p, segs in enumerate(path_segs):
        for s in segs:
            holders[s].append(p)
            row([(p, 1.0), (zoff + s, -1.0)], 0.0)                          # x_p <= z_s
    for s in range(n_s):
        row([(zoff + s, 1.0), *[(p, -1.0) for p in holders[s]]], 0.0)       # z_s <= sum x_p
    edge_cover: list[list[int]] = [[] for _ in range(n_e)]
    for s, es in enumerate(seg_edges):
        for e in es.tolist():
            edge_cover[e].append(s)
    for e in range(n_e):
        row([(yoff + e, 1.0), *[(zoff + s, -1.0) for s in edge_cover[e]]], 0.0)
    for s, (bidx, c) in enumerate(seg_disp):
        for b, cv in zip(bidx.tolist(), c.tolist(), strict=True):
            row([(zoff + s, float(cv)), (uoff + b, -1.0)], 0.0)             # c_b(s) z_s <= u_b

    a_ub = coo_matrix((vals, (rows, cols)), shape=(len(ub), n_var)).tocsr()
    obj = np.zeros(n_var)
    obj[yoff:uoff] = -edge_gain
    if length_price > 0.0 and edge_gain.size and seg_len.size:
        # Price metres in the OBJECTIVE rather than capping them. A hard metre cap has to be tuned
        # per block, and tuning it to match a baseline would beg the question the comparison asks.
        # A price is scale-free: `length_price` is normalized by the instance's own best available
        # gain-per-metre, so price 0 is the unpriced LP, price ~1 charges a segment roughly what an
        # average productive segment earns, and large prices buy only the very best metres.
        scale = float(edge_gain.sum()) / float(seg_len.sum())
        obj[zoff:yoff] += length_price * scale * seg_len
    lo = np.zeros(n_var)
    lo[uoff:] = base_c
    res = linprog(obj, A_ub=a_ub, b_ub=np.asarray(ub, dtype=float),
                  bounds=np.column_stack([lo, np.ones(n_var)]), method="highs")
    if not res.success:
        return np.zeros(n_s)
    return np.asarray(res.x[zoff:yoff], dtype=float)


@dataclass(frozen=True)
class ResistanceLPIdentity:
    substrate: Hashable
    max_displacement: float
    max_road_m: float
    chunks: int
    length_price: float


@dataclass
class ResistanceLPReblocker:
    """Choose the road set by LP against a displacement budget, re-linearizing between chunks."""

    substrate: Substrate = field(default_factory=ChordSubstrate)
    max_displacement: float = 0.10
    max_road_m: float = 1e6
    # Charge for road length in the objective. Lens A caps displacement and says nothing about
    # metres, so an unpriced optimizer keeps buying cheap parcel-boundary roads until the whole
    # fabric is meshed -- measured on the depth example: 42,937 m against clearance_looped's 9,878 m
    # for permeability 0.955. That is a correct solution to an under-specified problem. The price
    # makes the LP trade gain against metres instead. Normalized per instance (see
    # `solve_coverage_lp`), so one value is meaningful across blocks and regions.
    length_price: float = 0.0
    chunks: int = 8
    params: PermeabilityParams = field(default_factory=PermeabilityParams)

    @property
    def identity(self) -> ResistanceLPIdentity | None:
        if self.substrate.identity is None:
            return None
        return ResistanceLPIdentity(
            substrate=self.substrate.identity, max_displacement=float(self.max_displacement),
            max_road_m=float(self.max_road_m), chunks=int(self.chunks),
            length_price=float(self.length_price))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        graph = self.substrate.build(block)
        crs = block.crs
        empty = gpd.GeoDataFrame(geometry=[], crs=crs)
        geoms = list(block.parcels.geometry)
        if len(geoms) == 0 or len(graph.pts) == 0:
            return self._proposal(block, empty, {"roads": 0, "stopped": "empty"})

        adj = parcel_adjacency(geoms, self.params.corridor_m)
        r0 = _adaptive_r0(block, self.params)
        street = unary_union(list(block.streets.geometry))
        net0 = np.flatnonzero(
            shapely.dwithin(shapely.points(graph.pts), street, graph.net_tol)).tolist()
        if not net0:
            return self._proposal(block, empty, {"roads": 0, "stopped": "no street frontage"})

        pts = block.building_points
        n_b = len(pts)
        radii = building_radii(pts, CORRIDOR_M)
        disp_cap = self.max_displacement * n_b

        pt_tree = cKDTree(graph.pts)
        reps = np.array([[g.representative_point().x, g.representative_point().y]
                         for g in geoms])
        starts = pt_tree.query(reps)[1]
        w = np.concatenate([graph.edist, graph.edist])
        csr = csr_matrix(
            (w, (np.concatenate([graph.rows, graph.cols]),
                 np.concatenate([graph.cols, graph.rows]))),
            shape=(len(graph.pts), len(graph.pts)))
        ri, ci, dg, segs = _mesh(block, self.params, adj, r0)
        seg_tree = STRtree(list(segs)) if len(segs) else None

        best: list[LineString] = []
        best_perm = permeability(block, empty, self.params, adj=adj, r0=r0)
        base_c = np.zeros(n_b)
        k = max(self.chunks, 1)
        for t in range(k):
            built = gpd.GeoDataFrame(geometry=best, crs=crs) if best else empty
            _p, v = egress_power(block, built, self.params, adj=adj, r0=r0)
            corridor = _road_corridor(built, self.params.corridor_m)
            upgraded = (np.array([corridor.intersects(sg) for sg in segs], dtype=bool)
                        if corridor is not None and len(segs)
                        else np.zeros(len(segs), dtype=bool))
            edge_gain = linearized_gain(v, ri, ci, dg, upgraded)

            net = list(net0)
            for road in best:
                for x, y in road.coords:
                    net.extend(pt_tree.query_ball_point([x, y], graph.net_tol))
            net = list(dict.fromkeys(net))
            _d, pred, _s = dijkstra(csr, indices=net, return_predecessors=True, min_only=True)

            ids: dict[SegId, int] = {}
            seg_geom: list[LineString] = []
            path_segs: list[list[int]] = []
            for i in range(len(geoms)):
                if pred[starts[i]] < 0:
                    continue
                got = _path_segments(graph, pred, int(starts[i]), i, reps[i], street)
                idx = []
                for sid, geom in got:
                    if sid not in ids:
                        ids[sid] = len(seg_geom)
                        seg_geom.append(geom)
                    idx.append(ids[sid])
                if idx:
                    path_segs.append(idx)
            if not path_segs:
                break

            seg_len = np.array([g.length for g in seg_geom], dtype=float)
            seg_edges = [
                (seg_tree.query(g.buffer(self.params.corridor_m), predicate="intersects")
                 if seg_tree is not None else np.zeros(0, dtype=np.int64))
                for g in seg_geom
            ]
            seg_disp = segment_displacement(seg_geom, pts, radii, CORRIDOR_M)
            allow = disp_cap * (t + 1) / k
            spent_m = float(sum(r.length for r in best))
            z = solve_coverage_lp(path_segs, seg_len, seg_edges, seg_disp, edge_gain,
                                  n_b, base_c, allow, max(self.max_road_m - spent_m, 0.0),
                                  length_price=self.length_price)
            roads, base_c2 = self._round(path_segs, seg_len, seg_geom, seg_disp, z, base_c,
                                         allow, max(self.max_road_m - spent_m, 0.0))
            if not roads:
                continue
            trial = gpd.GeoDataFrame(geometry=[*best, *roads], crs=crs)
            perm = permeability(block, trial, self.params, adj=adj, r0=r0)
            if perm <= best_perm:
                continue
            best, best_perm, base_c = [*best, *roads], perm, base_c2

        gdf = gpd.GeoDataFrame(geometry=best, crs=crs)
        return self._proposal(block, gdf, {"roads": len(best),
                                           "permeability": float(best_perm)})

    def _round(
        self, path_segs: list[list[int]], seg_len: np.ndarray, seg_geom: list[LineString],
        seg_disp: list[tuple[np.ndarray, np.ndarray]], z: np.ndarray, base_c: np.ndarray,
        disp_budget: float, len_budget: float,
    ) -> tuple[list[LineString], np.ndarray]:
        """Commit paths in LP-confidence order, street-first, each segment at most once.

        Committing a PREFIX of a path is legal and is why `_path_segments` orders street-first: a
        prefix still reaches the street, it just stops short of the parcel. Whole-path-only
        rounding silently returns nothing whenever the instalment is smaller than the cheapest
        path. Displacement is tracked as the running per-building max, matching the union rule the
        metric uses, so a segment beside an already-displaced building is charged only the
        difference.
        """
        order = sorted(range(len(path_segs)),
                       key=lambda p: -float(min(z[s] for s in path_segs[p])))
        built: set[int] = set()
        out: list[LineString] = []
        cur = base_c.copy()
        spent = 0.0
        for p in order:
            for s in path_segs[p]:
                if s in built:
                    continue
                if spent + float(seg_len[s]) > len_budget:
                    break
                bidx, c = seg_disp[s]
                trial = cur.copy()
                if bidx.size:
                    trial[bidx] = np.maximum(trial[bidx], c)
                if float(trial.sum()) > disp_budget:
                    break
                built.add(s)
                cur = trial
                spent += float(seg_len[s])
                out.append(seg_geom[s])
        return out, cur

    def _proposal(self, block: Block, roads: gpd.GeoDataFrame,
                  params: dict[str, object]) -> Proposal:
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=(f"resistance_lp:{self.substrate.tag}:d{self.max_displacement:g}"
                         f":k{self.chunks}:lp{self.length_price:g}"),
            method="resistance_lp",
            params={**params, "substrate": self.substrate.tag,
                    "max_displacement": self.max_displacement, "chunks": self.chunks},
            block_identity=block.identity if self.identity is not None else None)
