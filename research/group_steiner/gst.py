"""Topology's objective solved as what it is: a GROUP STEINER TREE on the parcel-boundary graph.

Brelsford & Bettencourt's topological optimization: choose parcel-boundary edges to become road so
that every parcel has a corner on the road network, minimizing total new length. On topology's own
graph (`to_parcel_graph` + `_mark_streets_as_roads` + `define_interior_parcels`, exactly as
`TopologyMethod.propose` builds it):

  * the street network (every road node) is contracted to one ROOT;
  * each interior parcel is a GROUP: the set of its corners;
  * a solution is a tree from the root whose nodes hit every group; cost = its edge length.

Upper bound: shortest-path heuristic (attach the nearest unserved group from the whole tree) +
leaf pruning + MST-of-induced-subgraph improvement, restarted under LP-guided cost perturbations.
Lower bound: the LP relaxation of the directed-cut formulation (terminal t_g per group fed by
zero-cost arcs from the group's corners), with in-degree and flow-balance strengthening, cuts
separated by max-flow until none is violated. Every added cut is valid, so the LP value at ANY
round is a valid lower bound. Exact: the same formulation as a MILP, re-solved with cuts separated
on the integer solution until it is connected.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse.csgraph import connected_components, dijkstra, maximum_flow
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = 0


@dataclass
class Instance:
    n: int                              # nodes; 0 = the contracted street network
    eu: np.ndarray                      # undirected edges (u < v after contraction), int
    ev: np.ndarray
    ec: np.ndarray                      # length, m
    groups: list[np.ndarray]            # node ids of each interior parcel's corners
    xy: np.ndarray                      # node coordinates (root: nan), block CRS
    root_end: dict = field(default_factory=dict)  # (0, v) -> the real road-node xy it came from
    n_unreachable: int = 0              # groups dropped: no boundary path to the street at all
    node_groups: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_groups = [[] for _ in range(self.n)]
        for g, nodes in enumerate(self.groups):
            for v in nodes:
                self.node_groups[int(v)].append(g)

    def adjacency(self, cost: np.ndarray | None = None) -> sp.csr_matrix:
        c = self.ec if cost is None else cost
        # csgraph treats explicit zeros as absent edges: floor at a tiny positive weight
        c = np.maximum(c, 1e-9)
        a = sp.coo_matrix((np.r_[c, c], (np.r_[self.eu, self.ev], np.r_[self.ev, self.eu])),
                          shape=(self.n, self.n))
        return a.tocsr()


def build_instance(block, k: int = 0) -> Instance:
    """Topology's exact graph for `block`, streets contracted to node 0.

    `k` is the access target in parcels crossed: 0 = every parcel fronts a road (topology's
    universal access, peel depth 1); k = every parcel is within k shared-edge hops of one that
    does (peel depth k + 1). A face's group is then the corners of every face within k hops, and a
    group that contains another is dropped -- serving the smaller serves it."""
    from reblock.derive.parcel_graph import to_parcel_graph
    from reblock.methods.topology import _mark_streets_as_roads
    ppg = to_parcel_graph(block)
    g = ppg.graph
    # topology's `_streets_local_geometry` keeps only LineString rows, so a MultiLineString
    # street (a block with an inner ring) marks NO road at all; explode first.
    _mark_streets_as_roads(g, block.streets.explode(index_parts=False), ppg.origin)
    g.define_interior_parcels()
    ox, oy = ppg.origin
    nodes = list(g.G.nodes())
    idx: dict = {}
    xy = [(np.nan, np.nan)]
    for nd in nodes:
        if nd.road:
            idx[nd] = ROOT
        else:
            idx[nd] = len(xy)
            xy.append((nd.x + ox, nd.y + oy))
    best: dict[tuple[int, int], float] = {}
    root_end: dict[tuple[int, int], tuple[float, float]] = {}
    for a, b, d in g.G.edges(data=True):
        u, v = idx[a], idx[b]
        if u == v:
            continue                    # road-road edge: already built
        key = (min(u, v), max(u, v))
        w = float(d["weight"])
        if w < best.get(key, np.inf):
            best[key] = w
            if u == ROOT or v == ROOT:
                r = a if u == ROOT else b
                root_end[key] = (r.x + ox, r.y + oy)
    keys = sorted(best)
    eu = np.array([k[0] for k in keys], dtype=np.int64)
    ev = np.array([k[1] for k in keys], dtype=np.int64)
    ec = np.array([best[k] for k in keys])
    # Groups are built on the PEEL's own parcels and adjacency (`ParcelAdjacency`: a shared
    # boundary run within tolerance), so "within k parcels of a road" means exactly what the
    # peel scores. A parcel's corners are the graph nodes within tolerance of its boundary.
    import shapely
    from reblock.derive.access import STREET_TOL, ParcelAdjacency
    adj = ParcelAdjacency.of(block, STREET_TOL)
    node_list = [nd for nd in nodes]
    node_pts = shapely.points([(nd.x + ox, nd.y + oy) for nd in node_list])
    tree = shapely.STRtree(node_pts)
    pgeoms = np.asarray(block.parcels.geometry)
    pi, ni = tree.query(shapely.boundary(pgeoms), predicate="dwithin", distance=STREET_TOL)
    parcel_nodes: list[set[int]] = [set() for _ in pgeoms]
    for p_, n_ in zip(pi.tolist(), ni.tolist(), strict=True):
        parcel_nodes[p_].add(idx[node_list[n_]])
    sets: list[frozenset[int]] = []
    for i in range(len(pgeoms)):
        reach, frontier = {i}, {i}
        for _ in range(k):
            frontier = {b for a in frontier for b in adj.neighbours[a]} - reach
            reach |= frontier
        grp = frozenset().union(*(parcel_nodes[j] for j in reach))
        if grp and ROOT not in grp:
            sets.append(grp)
    sets = sorted(set(sets), key=len)
    kept: list[frozenset[int]] = []
    for s_ in sets:                          # drop supersets of an already-kept group
        if not any(q <= s_ for q in kept):
            kept.append(s_)
    groups = [np.array(sorted(s_), dtype=np.int64) for s_ in kept]
    # A group none of whose corners connects to the street through the boundary graph cannot be
    # served by any boundary road (topology crashes on these: NodeNotFound). Drop and count it.
    n = len(xy)
    a = sp.coo_matrix((np.ones(len(eu)), (eu, ev)), shape=(n, n))
    _, lab = connected_components(a, directed=False)
    reach = [gr for gr in groups if (lab[gr] == lab[ROOT]).any()]
    n_unreachable = len(groups) - len(reach)
    groups = reach
    return Instance(n=len(xy), eu=eu, ev=ev, ec=ec, groups=groups, xy=np.array(xy),
                    root_end=root_end, n_unreachable=n_unreachable)


# --- upper bound -------------------------------------------------------------------------------

def tree_cost(inst: Instance, edges: set[tuple[int, int]]) -> float:
    lut = {(int(u), int(v)): c for u, v, c in zip(inst.eu, inst.ev, inst.ec, strict=True)}
    return float(sum(lut[e] for e in edges))


def _served(inst: Instance, nodes: set[int]) -> np.ndarray:
    s = np.zeros(len(inst.groups), bool)
    for v in nodes:
        for g in inst.node_groups[v]:
            s[g] = True
    return s


def shortest_path_heuristic(inst: Instance, cost: np.ndarray | None = None
                            ) -> set[tuple[int, int]]:
    """Grow from the root: repeatedly attach, by a shortest path from ANY tree node, the unserved
    group whose nearest corner is closest. Returns the tree's undirected edges (u < v)."""
    adj = inst.adjacency(cost)
    tree_nodes = {ROOT}
    served = _served(inst, tree_nodes)
    edges: set[tuple[int, int]] = set()
    while not served.all():
        src = np.fromiter(tree_nodes, dtype=np.int64)
        dist, pred, _ = dijkstra(adj, indices=src, min_only=True, return_predecessors=True)
        best_g, best_v, best_d = -1, -1, np.inf
        for g in np.flatnonzero(~served):
            nodes = inst.groups[g]
            j = int(np.argmin(dist[nodes]))
            if dist[nodes[j]] < best_d:
                best_g, best_v, best_d = int(g), int(nodes[j]), float(dist[nodes[j]])
        if not np.isfinite(best_d):
            raise RuntimeError("a group is unreachable from the street")
        v = best_v
        while v not in tree_nodes:
            p = int(pred[v])
            edges.add((min(p, v), max(p, v)))
            tree_nodes.add(v)
            v = p
        served = _served(inst, tree_nodes)
    return edges


def prune(inst: Instance, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Drop leaves (never the root) whose groups are all served by another tree node."""
    edges = set(edges)
    while True:
        deg: dict[int, int] = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        count = np.zeros(len(inst.groups), np.int64)
        for v in deg:
            for g in inst.node_groups[v]:
                count[g] += 1
        removed = False
        for (u, v) in sorted(edges):
            for leaf in (u, v):
                if leaf != ROOT and deg.get(leaf) == 1 and all(
                        count[g] >= 2 for g in inst.node_groups[leaf]):
                    edges.discard((u, v))
                    for g in inst.node_groups[leaf]:
                        count[g] -= 1
                    deg[u] -= 1
                    deg[v] -= 1
                    removed = True
                    break
        if not removed:
            return edges


def mst_improve(inst: Instance, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """MST of the subgraph induced by the tree's nodes, then prune; repeat while it improves."""
    cur, cur_c = edges, tree_cost(inst, edges)
    while True:
        nodes = {ROOT} | {x for e in cur for x in e}
        keep = np.array([u in nodes and v in nodes for u, v in zip(inst.eu, inst.ev, strict=True)])
        sub = sp.coo_matrix((np.maximum(inst.ec[keep], 1e-9), (inst.eu[keep], inst.ev[keep])),
                            shape=(inst.n, inst.n)).tocsr()
        mst = minimum_spanning_tree(sub).tocoo()
        cand = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(mst.row, mst.col, strict=True)}
        cand = prune(inst, cand)
        c = tree_cost(inst, cand)
        if c < cur_c - 1e-9 and _served(inst, {ROOT} | {x for e in cand for x in e}).all() \
                and _connected_to_root(inst, cand):
            cur, cur_c = cand, c
        else:
            return cur


def _connected_to_root(inst: Instance, edges: set[tuple[int, int]]) -> bool:
    if not edges:
        return True
    u = np.array([e[0] for e in edges])
    v = np.array([e[1] for e in edges])
    a = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(inst.n, inst.n))
    _, lab = connected_components(a, directed=False)
    used = np.unique(np.r_[u, v])
    return bool((lab[used] == lab[ROOT]).all())


def improve(inst: Instance, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return mst_improve(inst, prune(inst, edges))


def upper_bound(inst: Instance, x_edge: np.ndarray | None = None, restarts: int = 20,
                seed: int = 0) -> tuple[set[tuple[int, int]], float, dict]:
    """SPH + improvement; then restarts under perturbed costs -- LP-guided when an LP edge
    solution is given (cost * (1 - x) + noise, the classic LP-driven rerouting), else random."""
    rng = np.random.default_rng(seed)
    best = improve(inst, shortest_path_heuristic(inst))
    best_c = tree_cost(inst, best)
    first = best_c
    if x_edge is not None:
        # LP-rounding starts: route only on the LP's support, and on costs discounted by x
        support = np.where(x_edge > 1e-6, inst.ec, inst.ec * 1e3)
        for cost in (support, inst.ec / (np.clip(x_edge, 0, 1) + 0.05)):
            try:
                cand = improve(inst, shortest_path_heuristic(inst, cost))
            except RuntimeError:
                continue
            c = tree_cost(inst, cand)
            if c < best_c - 1e-9:
                best, best_c = cand, c
    for r in range(restarts):
        if x_edge is not None and r % 2 == 0:
            cost = inst.ec * (1.0 - 0.9 * np.clip(x_edge, 0, 1)) * rng.uniform(0.95, 1.05, len(inst.ec))
        else:
            cost = inst.ec * rng.uniform(0.8, 1.2, len(inst.ec))
        cand = improve(inst, shortest_path_heuristic(inst, cost))
        c = tree_cost(inst, cand)
        if c < best_c - 1e-9:
            best, best_c = cand, c
    return best, best_c, {"sph_first": first}


# --- lower bound: directed-cut LP --------------------------------------------------------------

@dataclass
class Formulation:
    tail: np.ndarray                    # arcs: graph arcs both ways, then terminal arcs v -> t_g
    head: np.ndarray
    cost: np.ndarray
    n_graph_arcs: int
    n_nodes: int                        # graph nodes + one terminal per group
    terminals: np.ndarray               # node ids of t_g
    cuts: list[np.ndarray]              # arc-index sets, each sum >= 1


def formulation(inst: Instance) -> Formulation:
    tail = np.r_[inst.eu, inst.ev]
    head = np.r_[inst.ev, inst.eu]
    cost = np.r_[inst.ec, inst.ec]
    keep = head != ROOT                                  # nothing ever enters the root
    tail, head, cost = tail[keep], head[keep], cost[keep]
    na = len(tail)
    terms = inst.n + np.arange(len(inst.groups))
    tt, th = [], []
    for g, nodes in enumerate(inst.groups):
        tt.extend(nodes.tolist())
        th.extend([terms[g]] * len(nodes))
    tail = np.r_[tail, np.array(tt, dtype=np.int64)]
    head = np.r_[head, np.array(th, dtype=np.int64)]
    cost = np.r_[cost, np.zeros(len(tt))]
    f = Formulation(tail, head, cost, na, inst.n + len(inst.groups), terms, [])
    for t in terms:                                       # a terminal must be entered
        f.cuts.append(np.flatnonzero(head == t))
    return f


def _static_rows(f: Formulation) -> tuple[sp.csr_matrix, np.ndarray]:
    """A_ub x <= b_ub rows: in-degree <= 1 at every non-root node, and for every graph node that
    is not a terminal: each out-arc <= in-degree (flow balance), and in-degree <= out-degree (a
    minimal tree has no useless leaf; terminal arcs count as out-arcs)."""
    rows, cols, vals, b = [], [], [], []
    r = 0
    ins: dict[int, list[int]] = {}
    outs: dict[int, list[int]] = {}
    for a, (t, h) in enumerate(zip(f.tail, f.head, strict=True)):
        ins.setdefault(int(h), []).append(a)
        outs.setdefault(int(t), []).append(a)
    for v, arcs in ins.items():                           # in-degree <= 1
        rows += [r] * len(arcs); cols += arcs; vals += [1.0] * len(arcs); b.append(1.0); r += 1
    term_set = set(f.terminals.tolist())
    for v in range(1, f.n_nodes):
        if v in term_set:
            continue
        iv = ins.get(v, [])
        for a in outs.get(v, []):                         # x_a - x(in(v)) <= 0
            rows += [r] + [r] * len(iv); cols += [a] + iv; vals += [1.0] + [-1.0] * len(iv)
            b.append(0.0); r += 1
        ov = outs.get(v, [])
        if iv:                                            # x(in(v)) - x(out(v)) <= 0
            rows += [r] * (len(iv) + len(ov)); cols += iv + ov
            vals += [1.0] * len(iv) + [-1.0] * len(ov); b.append(0.0); r += 1
    a = sp.csr_matrix((vals, (rows, cols)), shape=(r, len(f.tail)))
    return a, np.array(b)


def _cut_rows(f: Formulation) -> tuple[sp.csr_matrix, np.ndarray]:
    rows, cols = [], []
    for i, arcs in enumerate(f.cuts):
        rows += [i] * len(arcs)
        cols += arcs.tolist()
    a = sp.csr_matrix((-np.ones(len(rows)), (rows, cols)), shape=(len(f.cuts), len(f.tail)))
    return a, -np.ones(len(f.cuts))


SCALE = 10**6
LP_METHOD = "highs-ds"


def separate(f: Formulation, x: np.ndarray, tol: float = 1e-4, max_new: int = 100000,
             nested: int = 8) -> list[np.ndarray]:
    """Max-flow root -> t_g on capacities x for every terminal. Each deficit yields the root-side
    min cut and the terminal-side back cut; with `nested` > 1 the cut's arcs are then saturated
    (capacity 1) and max-flow re-run, yielding up to `nested` further cuts for that terminal in
    the same round -- the standard remedy for the cut loop's tailing."""
    base_cap = np.floor(np.clip(x, 0, 1) * SCALE).astype(np.int64)
    new: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add(arcs: np.ndarray) -> None:
        key = arcs.tobytes()
        if len(arcs) and key not in seen:
            seen.add(key)
            new.append(arcs)

    for t in f.terminals:
        cap = base_cap.copy()
        for _ in range(max(nested, 1)):
            keep = cap > 0
            g = sp.csr_matrix((cap[keep].astype(np.int32), (f.tail[keep], f.head[keep])),
                              shape=(f.n_nodes, f.n_nodes))
            g.sum_duplicates()
            res = maximum_flow(g, ROOT, int(t))
            if res.flow_value >= (1.0 - tol) * SCALE:
                break
            resid = (g - res.flow).tocsr()
            resid.data = np.where(resid.data > 0, 1, 0)
            resid.eliminate_zeros()
            side = np.zeros(f.n_nodes, bool)
            side[sp.csgraph.breadth_first_order(resid, ROOT, directed=True,
                                                return_predecessors=False)] = True
            cut = np.flatnonzero(side[f.tail] & ~side[f.head])
            add(cut)
            tside = np.zeros(f.n_nodes, bool)
            tside[sp.csgraph.breadth_first_order(resid.T.tocsr(), int(t), directed=True,
                                                 return_predecessors=False)] = True
            back = np.flatnonzero(~tside[f.tail] & tside[f.head])
            add(back)
            cap[cut] = SCALE                              # saturate: find the NEXT cut
        if len(new) >= max_new:
            break
    return new


def lp_bound(inst: Instance, f: Formulation | None = None, max_rounds: int = 400,
             time_limit: float = 1800.0, verbose: bool = False, purge_after: int = 3,
             seed_cuts: list[np.ndarray] | None = None
             ) -> tuple[float, np.ndarray, Formulation, dict]:
    """Returns (LB, x per undirected edge = x_uv + x_vu, formulation, stats).

    Cuts slack for `purge_after` consecutive rounds leave the LP (the terminal in-cuts never do).
    Dropping a cut keeps the LP a relaxation, so every round's value is a valid bound and the
    best one is reported; purged cuts return if separation finds them violated again."""
    f = f or formulation(inst)
    n_fixed = len(f.terminals)
    if seed_cuts:
        f.cuts.extend(seed_cuts)
    a_static, b_static = _static_rows(f)
    slack_rounds = [0] * len(f.cuts)
    t0 = time.time()
    lb, best_x, x, rounds, exact = 0.0, None, np.zeros(len(f.tail)), 0, False
    while rounds < max_rounds and time.time() - t0 < time_limit:
        a_cut, b_cut = _cut_rows(f)
        res = linprog(f.cost, A_ub=sp.vstack([a_static, a_cut]).tocsr(),
                      b_ub=np.r_[b_static, b_cut], bounds=(0, 1), method=LP_METHOD)
        assert res.status == 0, res.message
        x = res.x
        if float(res.fun) >= lb:
            lb, best_x = float(res.fun), x
        rounds += 1
        t_lp = time.time()
        new = separate(f, x)
        # purge: cuts slack this round, for too long
        lhs = np.array([x[c].sum() for c in f.cuts])
        keep = []
        for i, c in enumerate(f.cuts):
            slack_rounds[i] = slack_rounds[i] + 1 if lhs[i] > 1.0 + 1e-6 else 0
            keep.append(i < n_fixed or slack_rounds[i] < purge_after)
        f.cuts = [c for c, k in zip(f.cuts, keep, strict=True) if k]
        slack_rounds = [r for r, k in zip(slack_rounds, keep, strict=True) if k]
        if verbose:
            print(f"    round {rounds}: lp {res.fun:.2f} best {lb:.2f} cuts {len(f.cuts)} "
                  f"+{len(new)} sep {time.time() - t_lp:.2f}s total {time.time() - t0:.1f}s",
                  flush=True)
        if not new:
            exact = True
            break
        f.cuts.extend(new)
        slack_rounds.extend([0] * len(new))
    x_edge = _edge_x(inst, f, best_x if best_x is not None else x)
    return lb, x_edge, f, {"lp_rounds": rounds, "lp_cuts": len(f.cuts), "lp_converged": exact,
                           "lp_s": time.time() - t0}


def _edge_x(inst: Instance, f: Formulation, x: np.ndarray) -> np.ndarray:
    lut = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(inst.eu, inst.ev, strict=True))}
    xe = np.zeros(len(inst.ec))
    for a in range(f.n_graph_arcs):
        u, v = int(f.tail[a]), int(f.head[a])
        xe[lut[(min(u, v), max(u, v))]] += x[a]
    return xe


def exact(inst: Instance, f: Formulation, time_limit: float = 600.0
          ) -> tuple[float | None, float, dict]:
    """See `exact_tree`; this returns only (optimum, bound, stats)."""
    opt, bound, st, _ = exact_tree(inst, f, time_limit)
    return opt, bound, st


def exact_tree(inst: Instance, f: Formulation, time_limit: float = 600.0
               ) -> tuple[float | None, float, dict, set[tuple[int, int]] | None]:
    """MILP on the formulation; cuts separated on each integer solution until it is a connected
    arborescence. Returns (optimum or None if not proven, best MILP dual bound, stats)."""
    a_static, b_static = _static_rows(f)
    t0 = time.time()
    rounds = 0
    bound = 0.0
    while time.time() - t0 < time_limit:
        a_cut, b_cut = _cut_rows(f)
        cons = LinearConstraint(sp.vstack([a_static, a_cut]).tocsr(), -np.inf, np.r_[b_static, b_cut])
        left = time_limit - (time.time() - t0)
        res = milp(f.cost, constraints=cons, integrality=np.ones(len(f.cost)),
                   bounds=Bounds(0, 1), options={"time_limit": max(left, 1.0), "disp": False})
        rounds += 1
        if res.x is None:
            return None, bound, {"milp_rounds": rounds, "milp_s": time.time() - t0}, None
        bound = max(bound, float(getattr(res, "mip_dual_bound", res.fun) or 0.0))
        if res.status != 0:                                # time limit: not proven
            return None, bound, {"milp_rounds": rounds, "milp_s": time.time() - t0}, None
        new = separate(f, np.round(res.x), tol=1e-6)
        if not new:
            chosen = np.flatnonzero(np.round(res.x[:f.n_graph_arcs]) > 0.5)
            tree = {(min(int(f.tail[a]), int(f.head[a])), max(int(f.tail[a]), int(f.head[a])))
                    for a in chosen}
            return float(res.fun), float(res.fun), {"milp_rounds": rounds,
                                                    "milp_s": time.time() - t0}, tree
        f.cuts.extend(new)
    return None, bound, {"milp_rounds": rounds, "milp_s": time.time() - t0}, None


def roads_gdf(inst: Instance, edges: set[tuple[int, int]], block, xy_of: dict | None = None):
    """The tree's NEW edges as LineStrings in the block CRS. Root-incident edges need the real
    road-node endpoint, which contraction dropped: `inst.root_end` supplies it."""
    import geopandas as gpd
    from shapely.geometry import LineString
    lines = []
    for u, v in sorted(edges):
        a = inst.xy[u] if u != ROOT else inst.root_end[(u, v)]
        b = inst.xy[v] if v != ROOT else inst.root_end[(u, v)]
        lines.append(LineString([tuple(a), tuple(b)]))
    from reblock.permeability import with_width
    return with_width(gpd.GeoDataFrame(geometry=lines, crs=block.crs), 7.0)


def mcf_bound(inst: Instance, method: str = "highs-ipm", time_limit: float = 1800.0
              ) -> tuple[float, np.ndarray, Formulation, dict]:
    """The same bound as the converged directed-cut LP, in ONE solve: a unit flow root -> t_g per
    group on its own copy of the arcs, each capped by x (multi-commodity flow; equal to the cut
    LP by max-flow/min-cut), plus the in-degree and balance rows on x. Returns (LB, x per
    undirected edge, formulation, stats)."""
    f = formulation(inst)
    t0 = time.time()
    na = len(f.tail)
    graph_arcs = np.arange(f.n_graph_arcs)
    a_static, b_static = _static_rows(f)
    # variable layout: x (na), then per group g: flow on graph arcs + g's terminal arcs
    blocks_arcs = []
    for g, t in enumerate(f.terminals):
        blocks_arcs.append(np.r_[graph_arcs, np.flatnonzero(f.head == t)])
    offsets = np.cumsum([na] + [len(b) for b in blocks_arcs])
    nvar = int(offsets[-1])
    eq_r, eq_c, eq_v, eq_b = [], [], [], []
    ub_r, ub_c, ub_v = [], [], []
    r_eq = 0
    r_ub = 0
    n_graph = inst.n
    for g, arcs in enumerate(blocks_arcs):
        base = int(offsets[g])
        cols = base + np.arange(len(arcs))
        t = int(f.terminals[g])
        # node ids local to this commodity: graph nodes 0..n-1, terminal -> n
        tl = f.tail[arcs]
        hd = np.where(f.head[arcs] == t, n_graph, f.head[arcs])
        # conservation: out - in = +1 at root, -1 at terminal, 0 elsewhere
        eq_r += (r_eq + tl).tolist() + (r_eq + hd).tolist()
        eq_c += cols.tolist() + cols.tolist()
        eq_v += [1.0] * len(arcs) + [-1.0] * len(arcs)
        bvec = np.zeros(n_graph + 1)
        bvec[ROOT], bvec[n_graph] = 1.0, -1.0
        eq_b.append(bvec)
        r_eq += n_graph + 1
        # linking: f_a - x_a <= 0
        ub_r += (r_ub + np.arange(len(arcs))).tolist() * 2
        ub_c += cols.tolist() + arcs.tolist()
        ub_v += [1.0] * len(arcs) + [-1.0] * len(arcs)
        r_ub += len(arcs)
    a_eq = sp.csr_matrix((eq_v, (eq_r, eq_c)), shape=(r_eq, nvar))
    a_link = sp.csr_matrix((ub_v, (ub_r, ub_c)), shape=(r_ub, nvar))
    a_stat = sp.hstack([a_static, sp.csr_matrix((a_static.shape[0], nvar - na))]).tocsr()
    cost = np.r_[f.cost, np.zeros(nvar - na)]
    t_build = time.time() - t0
    res = linprog(cost, A_ub=sp.vstack([a_link, a_stat]).tocsr(),
                  b_ub=np.r_[np.zeros(r_ub), b_static], A_eq=a_eq, b_eq=np.concatenate(eq_b),
                  bounds=(0, 1), method=method, options={"time_limit": time_limit})
    ok = res.status == 0
    x = res.x[:na] if res.x is not None else np.zeros(na)
    lb = float(res.fun) if ok else float("nan")
    return lb, _edge_x(inst, f, x), f, {"lp_s": time.time() - t0, "lp_build_s": t_build,
                                        "lp_vars": nvar, "lp_converged": ok,
                                        "lp_status": res.message[:80]}


def edge_displacement(inst: Instance, block, width_m: float = 7.0) -> np.ndarray:
    """Homes displaced by each edge ALONE: `buildings.displacement` of its own corridor, summed.
    The scored metric unions corridors, so a tree's true displacement is at most the sum of its
    edges' -- this over-charges edges sharing a node, slightly."""
    from shapely.geometry import LineString
    out = np.zeros(len(inst.ec))
    for i, (u, v) in enumerate(zip(inst.eu, inst.ev, strict=True)):
        a = inst.xy[u] if u != ROOT else inst.root_end[(int(u), int(v))]
        b = inst.xy[v] if v != ROOT else inst.root_end[(int(u), int(v))]
        out[i] = float(block.buildings.displacement(LineString([tuple(a), tuple(b)])
                                                    .buffer(width_m / 2.0)).sum())
    return out


def weighted(inst: Instance, extra: np.ndarray, lam: float) -> Instance:
    """The same instance with edge cost length + lam * extra (lam in metres per home)."""
    return Instance(n=inst.n, eu=inst.eu, ev=inst.ev, ec=inst.ec + lam * extra,
                    groups=inst.groups, xy=inst.xy, root_end=inst.root_end,
                    n_unreachable=inst.n_unreachable)


def _mst_tree(inst: Instance, nodes: set[int]) -> set[tuple[int, int]] | None:
    """MST of the subgraph induced by `nodes` (must contain the root), pruned; None if that
    subgraph does not connect every group-serving node to the root."""
    arr = np.fromiter(nodes, dtype=np.int64)
    mask = np.zeros(inst.n, bool)
    mask[arr] = True
    keep = mask[inst.eu] & mask[inst.ev]
    sub = sp.coo_matrix((np.maximum(inst.ec[keep], 1e-9), (inst.eu[keep], inst.ev[keep])),
                        shape=(inst.n, inst.n)).tocsr()
    _, lab = connected_components(sub, directed=False)
    comp = np.flatnonzero(lab == lab[ROOT])
    cmask = np.zeros(inst.n, bool)
    cmask[comp] = True
    if not _served(inst, set(comp.tolist())).all():
        return None
    k2 = keep & cmask[inst.eu]
    mst = minimum_spanning_tree(sp.coo_matrix((np.maximum(inst.ec[k2], 1e-9),
                                               (inst.eu[k2], inst.ev[k2])),
                                              shape=(inst.n, inst.n)).tocsr()).tocoo()
    return prune(inst, {(min(int(a), int(b)), max(int(a), int(b)))
                        for a, b in zip(mst.row, mst.col, strict=True)})


def vertex_insertion(inst: Instance, edges: set[tuple[int, int]], max_passes: int = 3
                     ) -> set[tuple[int, int]]:
    """Steiner-vertex insertion local search: for each node adjacent to the tree, add it, rebuild
    the MST over the enlarged node set, prune; keep any strict improvement. First-improvement,
    repeated for up to `max_passes` passes."""
    best, best_c = edges, tree_cost(inst, edges)
    adj: dict[int, list[int]] = {}
    for u, v in zip(inst.eu.tolist(), inst.ev.tolist(), strict=True):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    for _ in range(max_passes):
        improved = False
        nodes = {ROOT} | {x for e in best for x in e}
        cand = sorted({w for v in nodes for w in adj.get(v, []) if w not in nodes})
        for w in cand:
            if w in nodes:
                continue
            t = _mst_tree(inst, nodes | {w})
            if t is None:
                continue
            c = tree_cost(inst, t)
            if c < best_c - 1e-9:
                best, best_c = t, c
                nodes = {ROOT} | {x for e in best for x in e}
                improved = True
        if not improved:
            break
    return best


def reduced_cost_starts(inst: Instance, f, cbar: np.ndarray) -> list[np.ndarray]:
    """Undirected SPH costs from dual-ascent reduced costs: saturated arcs nearly free."""
    lut = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(inst.eu, inst.ev, strict=True))}
    red = np.full(len(inst.ec), np.inf)
    for a in range(f.n_graph_arcs):
        u, v = int(f.tail[a]), int(f.head[a])
        i = lut[(min(u, v), max(u, v))]
        red[i] = min(red[i], cbar[a])
    red[~np.isfinite(red)] = inst.ec[~np.isfinite(red)]
    return [red + 0.01 * inst.ec, red + 0.1 * inst.ec, 0.5 * red + 0.5 * inst.ec]


def fast_tree(inst: Instance, vi_passes: int = 2) -> tuple[set[tuple[int, int]], float, float, dict]:
    """No LP: round-robin dual ascent (bound + reduced costs), shortest-path heuristic from plain
    and reduced-cost starts, prune + MST, then vertex insertion. Returns (tree, ub, lb, stats)."""
    import da
    f = formulation(inst)
    t0 = time.time()
    lb, cbar = da.dual_ascent_rr(f)
    t_da = time.time() - t0
    best, best_c = None, np.inf
    for cost in [None, *reduced_cost_starts(inst, f, cbar)]:
        tr = improve(inst, shortest_path_heuristic(inst, cost))
        c = tree_cost(inst, tr)
        if c < best_c:
            best, best_c = tr, c
    t_sph = time.time() - t0 - t_da
    best = vertex_insertion(inst, best, max_passes=vi_passes)
    ub = tree_cost(inst, best)
    return best, ub, lb, {"da_s": t_da, "sph_s": t_sph, "vi_s": time.time() - t0 - t_da - t_sph}
