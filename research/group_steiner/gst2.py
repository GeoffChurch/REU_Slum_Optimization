"""Access with TWO independent exits: survivable group Steiner on the parcel-boundary graph.

Every interior k-group (the corners of a parcel and its k-hop neighbours, on the peel's own
adjacency) must reach the street by two EDGE-DISJOINT routes that leave through DIFFERENT street
sectors. Loops and through-routes are then forced by the requirement itself -- a dead end can never
give a second route -- instead of being bolted on afterwards.

Graph: node 0 = super-root R; nodes 1..S = street sectors (the street's road nodes, contracted per
`sector_m` of arc length); the rest are parcel corners. Decision edges are the boundary edges
(edges into a sector keep their real street endpoint for geometry). R -> sector links are fixed,
capacity 1, so two disjoint routes must use two sectors. A group already touching the street is
satisfied (the street is itself a connected network).

Heuristic: for each group, farthest first, the cheapest pair of edge-disjoint routes (Suurballe,
built edges free) is built; then reverse-delete: drop the costliest built edges whose removal leaves
every group with two disjoint routes inside the built network (max-flow check).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, maximum_flow

R = 0
EPS = 1e-9


@dataclass
class Inst2:
    n: int                              # 0 = R, 1..n_sectors = sectors, then corners
    n_sectors: int
    eu: np.ndarray
    ev: np.ndarray
    ec: np.ndarray                      # edge cost (length, or weighted)
    length: np.ndarray                  # true edge length, m
    groups: list[np.ndarray]
    xy: np.ndarray                      # node xy (R, sectors: nan)
    sector_end: dict = field(default_factory=dict)   # (u, v) with a sector end -> street xy
    n_unreachable: int = 0


def build(block, k: int = 1, sector_m: float = 150.0) -> Inst2:
    import shapely
    from shapely.ops import unary_union

    from reblock.derive.access import STREET_TOL, ParcelAdjacency
    from reblock.derive.parcel_graph import to_parcel_graph
    from reblock.methods.topology import _mark_streets_as_roads

    ppg = to_parcel_graph(block)
    g = ppg.graph
    streets = block.streets.explode(index_parts=False)
    _mark_streets_as_roads(g, streets, ppg.origin)
    g.define_interior_parcels()             # sets node.road from the edges just marked
    ox, oy = ppg.origin
    lines = [ln for ln in streets.geometry if ln.length > 0]
    # sector of a street node: which street line it lies on, and how far along it
    seg_counts = [max(1, int(round(ln.length / sector_m))) for ln in lines]
    offsets = np.cumsum([0] + seg_counts)
    n_sectors = int(offsets[-1])

    def sector_of(x: float, y: float) -> int:
        p = shapely.Point(x, y)
        d = [ln.distance(p) for ln in lines]
        j = int(np.argmin(d))
        pos = lines[j].project(p) / max(lines[j].length, 1e-9)
        return 1 + int(offsets[j]) + min(int(pos * seg_counts[j]), seg_counts[j] - 1)

    nodes = list(g.G.nodes())
    idx: dict = {}
    xy = [(np.nan, np.nan)] * (1 + n_sectors)
    for nd in nodes:
        if nd.road:
            idx[nd] = sector_of(nd.x + ox, nd.y + oy)
        else:
            idx[nd] = len(xy)
            xy.append((nd.x + ox, nd.y + oy))
    best: dict[tuple[int, int], tuple[float, tuple[float, float] | None]] = {}
    for a, b, d in g.G.edges(data=True):
        u, v = idx[a], idx[b]
        if u == v or (u <= n_sectors and v <= n_sectors):
            continue                    # along the street: already built
        key = (min(u, v), max(u, v))
        w = float(d["weight"])
        if w < best.get(key, (np.inf, None))[0]:
            end = None
            if u <= n_sectors or v <= n_sectors:
                r = a if u <= n_sectors else b
                end = (r.x + ox, r.y + oy)
            best[key] = (w, end)
    keys = sorted(best)
    eu = np.array([k_[0] for k_ in keys], dtype=np.int64)
    ev = np.array([k_[1] for k_ in keys], dtype=np.int64)
    ln_ = np.array([best[k_][0] for k_ in keys])
    sector_end = {k_: best[k_][1] for k_ in keys if best[k_][1] is not None}

    adj = ParcelAdjacency.of(block, STREET_TOL)
    node_pts = shapely.points([(nd.x + ox, nd.y + oy) for nd in nodes])
    tree = shapely.STRtree(node_pts)
    pgeoms = np.asarray(block.parcels.geometry)
    pi, ni = tree.query(shapely.boundary(pgeoms), predicate="dwithin", distance=STREET_TOL)
    parcel_nodes: list[set[int]] = [set() for _ in pgeoms]
    for p_, n_ in zip(pi.tolist(), ni.tolist(), strict=True):
        parcel_nodes[p_].add(idx[nodes[n_]])
    sets: set[frozenset[int]] = set()
    for i in range(len(pgeoms)):
        reach, frontier = {i}, {i}
        for _ in range(k):
            frontier = {b for a in frontier for b in adj.neighbours[a]} - reach
            reach |= frontier
        grp = frozenset().union(*(parcel_nodes[j] for j in reach))
        if grp and min(grp) > n_sectors:          # touches no sector: needs the two routes
            sets.add(grp)
    kept: list[frozenset[int]] = []
    for s_ in sorted(sets, key=len):
        if not any(q <= s_ for q in kept):
            kept.append(s_)
    del unary_union
    return Inst2(n=len(xy), n_sectors=n_sectors, eu=eu, ev=ev, ec=ln_.copy(), length=ln_,
                 groups=[np.array(sorted(s_), dtype=np.int64) for s_ in kept],
                 xy=np.array(xy), sector_end=sector_end)


def edge_homes(inst: Inst2, block, width_m: float = 7.0) -> np.ndarray:
    from shapely.geometry import LineString
    out = np.zeros(len(inst.eu))
    for i, (u, v) in enumerate(zip(inst.eu, inst.ev, strict=True)):
        a = inst.xy[u] if u > inst.n_sectors else inst.sector_end[(int(u), int(v))]
        b = inst.xy[v] if v > inst.n_sectors else inst.sector_end[(int(u), int(v))]
        out[i] = float(block.buildings.displacement(LineString([tuple(a), tuple(b)])
                                                    .buffer(width_m / 2.0)).sum())
    return out


# --- the augmented digraph used for one group ---------------------------------------------------

@dataclass
class _Arcs:
    tail: np.ndarray
    head: np.ndarray
    edge: np.ndarray        # decision edge id, or -1 for a fixed arc
    n: int                  # nodes incl. t (= n_nodes-2) and t2 (= n_nodes-1)


def _arcs(inst: Inst2, group: np.ndarray, usable: np.ndarray) -> _Arcs:
    """Decision edges (both directions, only `usable` ones), R -> each sector, and the group's
    corners -> t and -> t2 -> t (two parallel ways in, so both routes may end at one corner)."""
    t, t2 = inst.n, inst.n + 1
    ids = np.flatnonzero(usable)
    tail = [inst.eu[ids], inst.ev[ids], np.zeros(inst.n_sectors, np.int64),
            group, group, np.array([t2])]
    head = [inst.ev[ids], inst.eu[ids], np.arange(1, inst.n_sectors + 1), np.full(len(group), t),
            np.full(len(group), t2), np.array([t])]
    edge = [ids, ids, np.full(inst.n_sectors, -1), np.full(len(group), -1),
            np.full(len(group), -1), np.array([-1])]
    return _Arcs(np.concatenate(tail), np.concatenate(head), np.concatenate(edge), inst.n + 2)


def _path(pred: np.ndarray, src: int, dst: int) -> list[int] | None:
    if pred[dst] < 0 and dst != src:
        return None
    out = [dst]
    while out[-1] != src:
        out.append(int(pred[out[-1]]))
    return out[::-1]


def suurballe(inst: Inst2, group: np.ndarray, cost_e: np.ndarray, usable: np.ndarray
              ) -> set[int] | None:
    """Decision edges of the cheapest pair of edge-disjoint R -> group routes through different
    sectors (costs per decision edge; fixed arcs are free). None if no such pair exists."""
    A = _arcs(inst, group, usable)
    t = inst.n
    c = np.where(A.edge >= 0, cost_e[np.maximum(A.edge, 0)], 0.0) + EPS
    g1 = sp.csr_matrix((c, (A.tail, A.head)), shape=(A.n, A.n))
    d, pred = dijkstra(g1, indices=R, return_predecessors=True)
    p1 = _path(pred, R, t)
    if p1 is None:
        return None
    # arc lookup: (tail, head) -> index of the cheapest such arc
    first: dict[tuple[int, int], int] = {}
    for i, (a, b) in enumerate(zip(A.tail.tolist(), A.head.tolist(), strict=True)):
        j = first.get((a, b))
        if j is None or c[i] < c[j]:
            first[(a, b)] = i
    p1_arcs = [first[(a, b)] for a, b in zip(p1, p1[1:], strict=False)]
    p1_edges = {int(A.edge[i]) for i in p1_arcs if A.edge[i] >= 0}
    p1_fixed = {(int(A.tail[i]), int(A.head[i])) for i in p1_arcs if A.edge[i] < 0}
    # residual: drop P1's arcs AND the opposite arcs of P1's undirected edges; add reversals
    keep = np.ones(len(A.tail), bool)
    for i, e in enumerate(A.edge.tolist()):
        if e >= 0 and e in p1_edges:
            keep[i] = False
    for i in p1_arcs:
        if A.edge[i] < 0:
            keep[i] = False
    ok = keep & np.isfinite(d[A.tail]) & np.isfinite(d[A.head])
    red = c[ok] + d[A.tail[ok]] - d[A.head[ok]]
    tails, heads = list(A.tail[ok]), list(A.head[ok])
    costs = list(np.maximum(red, 0.0) + EPS)
    rev_edge: dict[tuple[int, int], int] = {}
    for i in p1_arcs:
        a, b = int(A.tail[i]), int(A.head[i])
        tails.append(b)
        heads.append(a)
        costs.append(EPS)
        rev_edge[(b, a)] = int(A.edge[i])
    g2 = sp.csr_matrix((costs, (tails, heads)), shape=(A.n, A.n))
    _, pred2 = dijkstra(g2, indices=R, return_predecessors=True)
    p2 = _path(pred2, R, t)
    if p2 is None:
        return None
    used = set(p1_edges)
    for a, b in zip(p2, p2[1:], strict=False):
        if (a, b) in rev_edge:                  # cancels a P1 arc
            e = rev_edge[(a, b)]
            if e >= 0:
                used.discard(e)
            continue
        # the real arc P2 took: a decision edge between a and b, if any
        e = _edge_between(inst, a, b)
        if e is not None:
            used.add(e)
    return used


_EDGE_LUT: dict[int, dict[tuple[int, int], int]] = {}


def _edge_between(inst: Inst2, a: int, b: int) -> int | None:
    lut = _EDGE_LUT.get(id(inst))
    if lut is None:
        lut = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(inst.eu, inst.ev, strict=True))}
        _EDGE_LUT[id(inst)] = lut
    return lut.get((min(a, b), max(a, b)))


def two_routes(inst: Inst2, group: np.ndarray, built: np.ndarray) -> bool:
    """Max-flow check: two edge-disjoint routes inside the built network (capacity 1 per edge
    per direction, sectors capacity 1, two ways into t)."""
    A = _arcs(inst, group, built)
    cap = np.ones(len(A.tail), np.int32)
    g = sp.csr_matrix((cap, (A.tail, A.head)), shape=(A.n, A.n))
    g.sum_duplicates()
    return maximum_flow(g, R, inst.n).flow_value >= 2


def solve(inst: Inst2, cost: np.ndarray | None = None, prune: bool = True
          ) -> tuple[np.ndarray, dict]:
    """Greedy Suurballe augmentation, farthest group first; then reverse-delete."""
    t0 = time.time()
    cost = inst.ec if cost is None else cost
    built = np.zeros(len(inst.eu), bool)
    everything = np.ones(len(inst.eu), bool)
    # order: farthest (single-route distance from R) first
    g_all = sp.csr_matrix((np.r_[cost, cost] + EPS, (np.r_[inst.eu, inst.ev], np.r_[inst.ev, inst.eu])),
                          shape=(inst.n, inst.n))
    src = np.arange(1, inst.n_sectors + 1)
    dist = dijkstra(g_all, indices=src, min_only=True)
    order = sorted(range(len(inst.groups)), key=lambda i: -float(np.min(dist[inst.groups[i]])))
    infeasible = 0
    pair_of: dict[int, set[int]] = {}
    for gi in order:
        grp = inst.groups[gi]
        if built.any() and two_routes(inst, grp, built):
            # already served by earlier roads: record WHICH pair, so pruning rechecks it
            pair_of[gi] = suurballe(inst, grp, np.zeros(len(cost)), built) or set()
            continue
        used = suurballe(inst, grp, np.where(built, 0.0, cost), everything)
        if used is None:
            infeasible += 1
            continue
        built[list(used)] = True
        pair_of[gi] = used
    t_aug = time.time() - t0
    removed = 0
    if prune:
        # reverse delete, costliest first; a removal must keep every group two-routed
        for e in sorted(np.flatnonzero(built), key=lambda e: -cost[e]):
            built[e] = False
            affected = [gi for gi, used in pair_of.items() if e in used]
            ok = True
            for gi in affected:
                if not two_routes(inst, inst.groups[gi], built):
                    ok = False
                    break
            if ok:
                removed += 1
                for gi in affected:          # re-record a surviving pair inside the built network
                    pair_of[gi] = suurballe(inst, inst.groups[gi], np.zeros(len(cost)),
                                            built) or set()
            else:
                built[e] = True
    return built, {"infeasible_groups": infeasible, "aug_s": t_aug,
                   "prune_s": time.time() - t0 - t_aug, "pruned": removed}


def roads(inst: Inst2, built: np.ndarray, block):
    import geopandas as gpd
    from shapely.geometry import LineString

    from reblock.permeability import with_width
    lines = []
    for e in np.flatnonzero(built):
        u, v = int(inst.eu[e]), int(inst.ev[e])
        a = inst.xy[u] if u > inst.n_sectors else inst.sector_end[(u, v)]
        b = inst.xy[v] if v > inst.n_sectors else inst.sector_end[(u, v)]
        lines.append(LineString([tuple(a), tuple(b)]))
    return with_width(gpd.GeoDataFrame(geometry=lines, crs=block.crs), 7.0)


# --- corner mode and the redundancy prize ----------------------------------------------------------

def routes(inst: Inst2, target: np.ndarray, built: np.ndarray) -> int:
    """Max number of edge-disjoint R -> target routes inside the built network (0, 1 or 2)."""
    A = _arcs(inst, target, built)
    g = sp.csr_matrix((np.ones(len(A.tail), np.int32), (A.tail, A.head)), shape=(A.n, A.n))
    g.sum_duplicates()
    return int(maximum_flow(g, R, inst.n).flow_value)


def _single(inst: Inst2, target: np.ndarray, cost_e: np.ndarray) -> set[int] | None:
    """Decision edges of the cheapest single R -> target route."""
    A = _arcs(inst, target, np.ones(len(inst.eu), bool))
    c = np.where(A.edge >= 0, cost_e[np.maximum(A.edge, 0)], 0.0) + EPS
    g = sp.csr_matrix((c, (A.tail, A.head)), shape=(A.n, A.n))
    _, pred = dijkstra(g, indices=R, return_predecessors=True)
    p = _path(pred, R, inst.n)
    if p is None:
        return None
    out = set()
    for a, b in zip(p, p[1:], strict=False):
        e = _edge_between(inst, a, b)
        if e is not None:
            out.add(e)
    return out


def solve2(inst: Inst2, cost: np.ndarray, mode: str = "corner", prize: float | None = None,
           n_corners: int = 4) -> tuple[np.ndarray, dict]:
    """mode 'corner': a group is served by two disjoint routes ending at ONE of its corners (a
    loop through the parcel), trying its `n_corners` nearest corners. `prize` = None makes that
    mandatory for every group; a number gives every group one route and adds the second only if it
    costs at most `prize` more (the redundancy prize). Farthest group first; reverse-delete after."""
    t0 = time.time()
    built = np.zeros(len(inst.eu), bool)
    g_all = sp.csr_matrix((np.r_[cost, cost] + EPS, (np.r_[inst.eu, inst.ev],
                                                     np.r_[inst.ev, inst.eu])), shape=(inst.n, inst.n))
    dist = dijkstra(g_all, indices=np.arange(1, inst.n_sectors + 1), min_only=True)
    order = sorted(range(len(inst.groups)), key=lambda i: -float(np.min(dist[inst.groups[i]])))
    need: dict[int, tuple[np.ndarray, int]] = {}      # group -> (target corners, routes required)
    uses: dict[int, set[int]] = {}                     # group -> edges of its recorded route(s)
    infeasible = 0
    n_pairs = 0

    def price(used: set[int]) -> float:
        return float(sum(cost[e] for e in used if not built[e]))

    for gi in order:
        grp = inst.groups[gi]
        corners = grp[np.argsort(dist[grp])][:n_corners]
        # already served?
        if built.any():
            served = [v for v in corners if routes(inst, np.array([v]), built) >= 2]
            if served:
                need[gi] = (corners, 2)       # any of its nearest corners will do
                uses[gi] = suurballe(inst, np.array([served[0]]), np.zeros(len(cost)), built) or set()
                continue
        best_pair, best_v = None, None
        for v in corners:
            used = suurballe(inst, np.array([v]), np.where(built, 0.0, cost),
                             np.ones(len(inst.eu), bool))
            if used is not None and (best_pair is None or price(used) < price(best_pair)):
                best_pair, best_v = used, v
        if prize is None:
            if best_pair is None:
                infeasible += 1
                continue
            built[list(best_pair)] = True
            need[gi] = (corners, 2)
            uses[gi] = set(best_pair)
            n_pairs += 1
            continue
        one = _single(inst, grp, np.where(built, 0.0, cost))
        if one is None:
            infeasible += 1
            continue
        if best_pair is not None and price(best_pair) - price(one) <= prize:
            built[list(best_pair)] = True
            need[gi] = (corners, 2)
            uses[gi] = set(best_pair)
            n_pairs += 1
        else:
            built[list(one)] = True
            need[gi] = (grp, 1)
            uses[gi] = set(one)
    t_aug = time.time() - t0

    def ok(gi: int) -> bool:
        tgt, k = need[gi]
        if k == 1:
            return routes(inst, tgt, built) >= 1
        return any(routes(inst, np.array([v]), built) >= 2 for v in tgt)

    removed = 0
    # reverse delete: a group whose recorded routes avoid edge e keeps them, so only the groups
    # whose routes USE e are rechecked; after a removal their routes are re-recorded
    users: dict[int, set[int]] = {}
    for gi, used in uses.items():
        for e in used:
            users.setdefault(e, set()).add(gi)
    for e in sorted(np.flatnonzero(built), key=lambda e: -cost[e]):
        built[e] = False
        cand = users.get(int(e), set())
        if all(ok(gi) for gi in cand):
            removed += 1
            for gi in list(cand):
                tgt, k = need[gi]
                new_used = (_single(inst, tgt, np.where(built, 0.0, np.inf)) if k == 1 else
                            next((u for v in tgt if (u := suurballe(inst, np.array([v]),
                                  np.where(built, 0.0, np.inf), built)) is not None), None))
                for x in uses.get(gi, ()):
                    users.get(x, set()).discard(gi)
                uses[gi] = new_used or set()
                for x in uses[gi]:
                    users.setdefault(x, set()).add(gi)
        else:
            built[e] = True
    return built, {"infeasible_groups": infeasible, "pairs": n_pairs, "groups": len(need),
                   "aug_s": t_aug, "prune_s": time.time() - t0 - t_aug, "pruned": removed}


_INC: dict[int, dict[int, list[int]]] = {}


def _incident(inst: Inst2, x: int) -> list[int]:
    inc = _INC.get(id(inst))
    if inc is None:
        inc = {}
        for i, (u, v) in enumerate(zip(inst.eu.tolist(), inst.ev.tolist(), strict=True)):
            inc.setdefault(u, []).append(i)
            inc.setdefault(v, []).append(i)
        _INC[id(inst)] = inc
    return inc.get(x, [])
