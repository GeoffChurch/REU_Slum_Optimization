"""Our Demo's methods scored on topology's own two objectives (owner 2026-11-04: "how we compare
with them on their own terms"), on one block (the Demo's face-off block, 41785).

Their objectives (Brelsford, Martin & Bettencourt, EPB 2019 doi 10.1177/2399808317712715, the
"Topological optimization", "Geometric optimization" and Results sections; Sci. Adv. 2018
eaar4644, supplement Sections SD and SE):
  1. topological: the least new road (lines along parcel boundaries, width ignored) that makes
     the block universally accessible, every parcel touching a road; scored as new road length
     (the best of many sampled runs is their answer);
  2. geometric: from that solution, greedy "bisecting" through-paths that lower the mean
     parcel-to-parcel travel distance T = sum_ij T_ij / n^2, T_ij the shortest distance between
     parcels i and j along roads and paths only; scored as T against total road length, for up to
     five or six bisections.

One travel function (`travel`) applied identically to every network. A network is its new roads'
centrelines and their paved area (`Net`). The graph (`built`) is the street (the block's boundary
ring, Block.streets) and the new centrelines, noded, with near misses joined at STREET_TOL
(reblock.derive.access's seam tolerance: ends within it merge, an end within it of a segment
splits that segment there; each parcel vertex within it of a road becomes a node too, as the
parcel corners are nodes in their graph), then every edge cut into SEG_M pieces (nodes that only
attach parcels). Their conventions, from ext/topology's
shortest_path_p2p_matrix (full=False, travelcost=False): road-only travel; road nodes within
CULDESAC_M joined by straight edges (their __road_connections_through_culdesac; among road nodes
at parcel corners, the nodes their graph has, not every densified metre); a parcel tied at
zero cost to every road node it touches (their centroid's fake edges to its corners), so T_ij is
the least road distance between any access point of i and any of j; the mean over the n(n-1)
ordered pairs i != j (their code's; the papers write n^2, 0.16% apart at n = 638). Unreachable
pairs are counted, never dropped silently: T is the mean over reachable pairs, `unreached` their
count.

Conventions (Convention, one row each):
  line      theirs: a parcel touches a road within STREET_TOL of its centreline (paths are lines);
  corridor  ours: a parcel touches a road where it comes within the road's paved half-width of a
            centreline node (its distance to the paved area's edge there), i.e. touches the paved
            corridor -- what a 7 m road reaches;
  and with `walk` (their travelcost=True, full=True): travel also along every parcel boundary at
  WALK_X times its length, which gives a T for networks short of universal access.
Their published T is `line` road-only.

Access (objective 1): a parcel is served when it touches a road in the street's component;
`peel` is the deepest BFS peel layer (reblock.derive.access's parcel adjacency at STREET_TOL,
1 = every parcel served, our k analogue).

The bisection stage (`bisect`), their code's rule (bisecting_path_endpoints, build_path): among
the road nodes that are parcel corners, the pair whose road distance (road-only, culs-de-sac
joined) over their straight-line distance is largest; the shortest path between them along parcel
boundaries and roads (all at their length), trimmed of road at both ends (shorten_path), built.
Their papers' wording (the parcel p* minimizing G_p / N_p) is not what the code does; the code is
what produced their figures. New length counts only the new edges (their bisecting_road counts
the trimmed path whole, road crossed in its middle too).

Networks (`nets`), each in its own build order, as the Demo cuts them:
  topology         the kept run (alpha 10, seed 0; atlas/topology_run.py), in the order its watched
                   re-run built it (topology_built_roads.parquet), cut at its paths' ends; then its
                   bisection stage;
  topology.a*.s*   the other samples (runs.py), whole, the best of them bisected too;
  steiner          the group Steiner floor (floor.py), whole, then bisected;
  shock            Shock Reblock (the ears run COMPARED.ours names), each step's corridor
                   (networks.Grown's region); its centreline the corridor's skeleton
                   (networks.skeleton's raster, simplified by SKEL_SIMPLIFY), each end within the
                   corridor of the street joined to it;
  cycle_native     the Loop Network (runs.py), its canonical order's prefixes.
Our lenses where the build-out rows have them (the grid tables' D and L_P at the same stops);
cycle_native's and the floor's scored here (`lenses`, as topology_run.py scores a network).

    T=research/roadless/atlas/topology_terms/terms.py    # from the repo root, PYTHONPATH=.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        PYTHONDONTWRITEBYTECODE=1 uv run python $T rows <block> <part>   # rows/<part>.parquet
    ... $T lenses <block> cycle_native|trees   # lenses.json (their D and L_P)
    ... $T cross <block>        # crosscheck.json against crosscheck.py's matrix
    ... $T report <block>       # tables, curves.png
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: common, road_rescore
sys.path.insert(0, str(HERE.parent))               # atlas: block_site
sys.path.insert(0, str(HERE.parent / "roads5810"))  # networks (imported where used: heavy)
import common  # noqa: E402
import road_rescore as rr  # noqa: E402
from block_site import Site  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402
from reblock.derive.access import STREET_TOL, ParcelAdjacency  # noqa: E402
from reblock.permeability import with_width  # noqa: E402

RUNS = HERE / "runs"
ROWS = HERE / "rows"            # rows/<part>.parquet
SEG_M = 1.0              # m: edges cut to this after noding (attachment sees every metre)
CULDESAC_M = 5.0         # m: their __road_connections_through_culdesac threshold
WALK_X = 10.0            # their travelcost=True factor on non-road edges
SKEL_SIMPLIFY = 0.5      # m: the corridor skeleton's raster stairs simplified away
ROAD_W = 7.0             # m: the line methods' road width (topology's, the floor's, bisections')
BISECTIONS = 6
SHOCK = "ears.paths.b20.f100.e30.loop.nodt.gl0.25Croad.n2s0.1.M8.K4.D0.91.v3"  # build.py COMPARED
TOPOLOGY_GRID = "topology.a10.s0.w7.built"                                     # its grid table
DEMO_D = (0.05, 0.10, 0.15)                        # the Demo page's slider budgets reported


@dataclass(frozen=True)
class Convention:
    """How a parcel meets a road, and whether travel may leave the roads."""
    name: str
    corridor: bool               # attach within the paved half-width (else within STREET_TOL)
    walk: bool                   # parcel boundaries walkable at WALK_X (else roads only)


CONVENTIONS = (Convention("line", corridor=False, walk=False),
               Convention("corridor", corridor=True, walk=False),
               Convention("line+walk", corridor=False, walk=True),
               Convention("corridor+walk", corridor=True, walk=True))
THEIRS = CONVENTIONS[0]


class Net(NamedTuple):
    """A network's state: its new roads' centrelines and paved area (block CRS)."""
    method: str
    stop: int                    # prefix length in the method's own units (paths, steps, roads)
    lines: np.ndarray            # LineStrings
    paved: Any                   # Polygon / MultiPolygon (empty for none)
    stage: str                   # "build" | "bisect"


class Ctx(NamedTuple):
    """The block's fixed parts."""
    b: Any
    parcels: np.ndarray          # Polygons
    ptree: shapely.STRtree
    pverts: np.ndarray           # (k, 2) every parcel exterior vertex
    pboundary: np.ndarray        # (m, 2, 2) parcel boundary segments, each once
    street: np.ndarray           # LineStrings
    adj: list[set[int]]


def ctx_of(b: Any) -> Ctx:
    parcels = np.array(b.parcels.geometry, dtype=object)
    xy, idx = shapely.get_coordinates(shapely.get_exterior_ring(parcels), return_index=True)
    same = idx[1:] == idx[:-1]
    segs = np.stack([xy[:-1][same], xy[1:][same]], axis=1)
    key = np.round(np.sort(segs.reshape(-1, 4).reshape(-1, 2, 2), axis=1).reshape(-1, 4), 6)
    _, first = np.unique(key, axis=0, return_index=True)
    return Ctx(b=b, parcels=parcels, ptree=shapely.STRtree(parcels),
               pverts=np.unique(np.round(xy, 9), axis=0), pboundary=segs[np.sort(first)],
               street=shapely.get_parts(np.array(b.streets.geometry, dtype=object)),
               adj=ParcelAdjacency.of(b, STREET_TOL).neighbours)


# ------------------------------------------------------------------ the graph

class Graph(NamedTuple):
    xy: np.ndarray               # (V, 2)
    eu: np.ndarray
    ev: np.ndarray
    ew: np.ndarray               # m
    road: np.ndarray             # per edge: a road (street, new road, cul-de-sac join)
    join: np.ndarray             # per edge: a cul-de-sac join (travel only, never built on)
    corner: np.ndarray           # per node: a parcel vertex lies on it
    mxy: np.ndarray              # (M, 2) every point merged into a node
    mlab: np.ndarray             # (M,) its node
    street: np.ndarray           # per node: on the street


def segments(lines: np.ndarray) -> np.ndarray:
    """(k, 2, 2) the lines' segments."""
    if len(lines) == 0:
        return np.zeros((0, 2, 2))
    xy, idx = shapely.get_coordinates(shapely.get_parts(lines), return_index=True)
    same = idx[1:] == idx[:-1]
    return np.stack([xy[:-1][same], xy[1:][same]], axis=1)


def _project(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    L2 = (d * d).sum(axis=1)
    return np.clip(((p - a) * d).sum(axis=1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)


def built(ctx: Ctx, lines: np.ndarray, *, walk: bool) -> Graph:
    """The street and `lines` (and with `walk` the parcel boundaries) as one noded graph (module
    docstring), road-only edges flagged, cul-de-sac joins among road nodes added."""
    road = np.concatenate([segments(ctx.street), segments(lines)])
    n_street = len(segments(ctx.street))
    allseg = np.concatenate([road, ctx.pboundary]) if walk else road
    pieces = shapely.get_parts(shapely.node(shapely.multilinestrings(shapely.linestrings(allseg))))
    xy, idx = shapely.get_coordinates(pieces, return_index=True)
    same = idx[1:] == idx[:-1]
    seg = np.stack([xy[:-1][same], xy[1:][same]], axis=1)
    mid = seg.mean(axis=1)
    rtree = shapely.STRtree(shapely.linestrings(road))
    hit = rtree.query(shapely.points(mid), predicate="dwithin", distance=1e-7)
    is_road = np.zeros(len(seg), bool)
    is_road[hit[0]] = True
    on_street = np.zeros(len(seg), bool)
    on_street[hit[0][hit[1] < n_street]] = True
    # near misses: segment ends and parcel vertices within STREET_TOL of a segment's interior
    ends = np.unique(np.round(seg.reshape(-1, 2), 9), axis=0)
    cand = np.concatenate([ends, ctx.pverts])
    stree = shapely.STRtree(shapely.linestrings(seg))
    pi, si = stree.query(shapely.points(cand), predicate="dwithin", distance=STREET_TOL)
    a, bb = seg[si, 0], seg[si, 1]
    t = _project(cand[pi], a, bb)
    L = np.linalg.norm(bb - a, axis=1)
    inner = (t * L > 1e-7) & ((1 - t) * L > 1e-7)
    splits: dict[int, list[float]] = {}
    for s_, t_ in zip(si[inner].tolist(), t[inner].tolist(), strict=True):
        splits.setdefault(s_, []).append(t_)
    out_a, out_b, out_road, out_street = [], [], [], []
    keep = np.ones(len(seg), bool)
    for s_, ts in splits.items():
        keep[s_] = False
        ts.sort()
        tt = np.array([0.0, *ts, 1.0])
        pts = seg[s_, 0] + tt[:, None] * (seg[s_, 1] - seg[s_, 0])
        out_a.append(pts[:-1])
        out_b.append(pts[1:])
        out_road.append(np.full(len(pts) - 1, is_road[s_]))
        out_street.append(np.full(len(pts) - 1, on_street[s_]))
    A = np.concatenate([seg[keep, 0], *out_a])
    B = np.concatenate([seg[keep, 1], *out_b])
    R = np.concatenate([is_road[keep], *out_road])
    S = np.concatenate([on_street[keep], *out_street])
    # nodes: every end, merged within STREET_TOL
    P = np.concatenate([A, B])
    u_xy, inv = np.unique(np.round(P, 9), axis=0, return_inverse=True)
    inv = inv.ravel()
    pairs = cKDTree(u_xy).query_pairs(STREET_TOL, output_type="ndarray")
    m = len(u_xy)
    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m)) \
        if len(pairs) else sp.coo_matrix((m, m))
    _, lab = connected_components(g, directed=False)
    nid = lab[inv]
    V = int(lab.max()) + 1
    first = np.full(V, -1)
    first[lab[::-1]] = np.arange(m)[::-1]
    nxy = u_xy[first]
    eu, ev = nid[:len(A)], nid[len(A):]
    ew = np.linalg.norm(B - A, axis=1)
    ok = eu != ev
    eu, ev, ew, R, S = eu[ok], ev[ok], ew[ok], R[ok], S[ok]
    corner = np.zeros(V, bool)                 # a node holding a parcel vertex (merged into it)
    d, _ = cKDTree(ctx.pverts).query(u_xy, distance_upper_bound=STREET_TOL)
    corner[lab[np.isfinite(d)]] = True
    # every edge cut into pieces of at most SEG_M, the cut points nodes of their own (never merged:
    # merging densified points within STREET_TOL of one another made shortcuts at near misses,
    # 1.7% of their T on the kept network), so a parcel or a paved half-width meets a node
    k = np.maximum(np.ceil(ew / SEG_M).astype(int), 1)
    longe = np.flatnonzero(k > 1)
    if len(longe):
        reps = k[longe] - 1
        e_of = np.repeat(longe, reps)
        frac = np.concatenate([np.arange(1, r + 1) / (r + 1) for r in reps])
        pa, pb = nxy[eu[e_of]], nxy[ev[e_of]]
        new_xy = pa + frac[:, None] * (pb - pa)
        new_id = V + np.arange(len(e_of))
        chain_u, chain_v, chain_e = [], [], []
        pos = 0
        for e, r in zip(longe.tolist(), reps.tolist(), strict=True):
            ids = np.r_[eu[e], new_id[pos:pos + r], ev[e]]
            chain_u.append(ids[:-1])
            chain_v.append(ids[1:])
            chain_e.append(np.full(r + 1, e))
            pos += r
        ce = np.concatenate(chain_e)
        short = np.flatnonzero(k == 1)
        eu = np.r_[eu[short], np.concatenate(chain_u)]
        ev = np.r_[ev[short], np.concatenate(chain_v)]
        ew = np.r_[ew[short], ew[ce] / k[ce]]
        R, S = np.r_[R[short], R[ce]], np.r_[S[short], S[ce]]
        nxy = np.concatenate([nxy, new_xy])
        corner = np.r_[corner, np.zeros(len(new_xy), bool)]
        u_xy = np.concatenate([u_xy, new_xy])
        lab = np.r_[lab, new_id]
        V += len(new_xy)
    street_node = np.zeros(V, bool)
    street_node[eu[S]] = True
    street_node[ev[S]] = True
    road_node = np.zeros(V, bool)
    road_node[eu[R]] = True
    road_node[ev[R]] = True
    # joined among the road nodes their graph has (parcel corners), not the densified points:
    # joining every metre of road to every other within CULDESAC_M shortened their T by 10%
    rn = np.flatnonzero(road_node & corner)
    jp = cKDTree(nxy[rn]).query_pairs(CULDESAC_M, output_type="ndarray")
    ju, jv = rn[jp[:, 0]], rn[jp[:, 1]]
    jw = np.linalg.norm(nxy[ju] - nxy[jv], axis=1)
    return Graph(xy=nxy, eu=np.r_[eu, ju], ev=np.r_[ev, jv], ew=np.r_[ew, jw],
                 road=np.r_[R, np.ones(len(ju), bool)], join=np.r_[np.zeros(len(eu), bool),
                                                                   np.ones(len(ju), bool)],
                 corner=corner, street=street_node, mxy=u_xy, mlab=lab)


def csr(g: Graph, w: np.ndarray, keep: np.ndarray) -> sp.csr_matrix:
    """Undirected CSR of the kept edges at weights `w`, parallel edges at their least weight."""
    u, v, ww = g.eu[keep], g.ev[keep], np.maximum(w[keep], 1e-9)
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    df = pd.DataFrame(dict(u=lo, v=hi, w=ww)).groupby(["u", "v"], sort=False).w.min().reset_index()
    n = len(g.xy)
    a = sp.coo_matrix((np.r_[df.w, df.w], (np.r_[df.u, df.v], np.r_[df.v, df.u])), shape=(n, n))
    return a.tocsr()


# ------------------------------------------------------------------ travel and access

class Travel(NamedTuple):
    T: float                     # mean over reachable ordered pairs i != j, m
    unreached: int               # ordered pairs i != j with no route
    pairs: int                   # n(n-1)
    served: np.ndarray           # per parcel: touches a road in the street's component
    peel: float                  # deepest peel layer (inf: a parcel no peel reaches)
    matrix: np.ndarray           # (n, n) T_ij, inf unreachable


def halfwidth(g: Graph, paved: Any, corridor: bool) -> np.ndarray:
    """Per merged point (Graph.mxy) the paved half-width it reaches across (0 under `line`)."""
    if not corridor or paved is None or shapely.is_empty(paved):
        return np.zeros(len(g.mxy))
    pts = shapely.points(g.mxy)
    inside = shapely.covers(paved, pts)
    hw = np.zeros(len(g.mxy))
    hw[inside] = shapely.distance(pts[inside], shapely.boundary(paved))
    return hw


def travel(ctx: Ctx, net: Net, conv: Convention, *, g: Graph | None = None) -> Travel:
    """T and access of `net` under `conv` (module docstring)."""
    g = g if g is not None else built(ctx, net.lines, walk=conv.walk)
    w = np.where(g.road, g.ew, g.ew * WALK_X)
    adj = csr(g, w, np.ones(len(g.ew), bool) if conv.walk else g.road)
    road_adj = csr(g, g.ew, g.road)
    hw = halfwidth(g, net.paved, conv.corridor)
    road_node = np.zeros(len(g.xy), bool)
    road_node[g.eu[g.road]] = True
    road_node[g.ev[g.road]] = True
    _, lab = connected_components(road_adj, directed=False)
    street_lab = set(lab[g.street].tolist())
    reach_street = np.isin(lab, list(street_lab)) & road_node
    pts = shapely.points(g.mxy)
    # road nodes attach within their half-width; walk nodes (parcel corners) to their own parcels;
    # tested at every point merged into the node, so the merge never moves a parcel out of reach
    mi, pi = ctx.ptree.query(pts, predicate="dwithin", distance=hw + STREET_TOL)
    ni = g.mlab[mi]
    ok = road_node[ni]
    if conv.walk:
        ok |= shapely.distance(pts[mi], ctx.parcels[pi]) <= STREET_TOL
    pair = np.unique(np.stack([ni[ok], pi[ok]], axis=1), axis=0)
    ni, pi = pair[:, 0], pair[:, 1]
    n = len(ctx.parcels)
    served = np.zeros(n, bool)
    served[pi[reach_street[ni]]] = True
    order = np.lexsort((ni, pi))
    ni, pi = ni[order], pi[order]
    has = np.zeros(n, bool)
    has[pi] = True
    starts = np.searchsorted(pi, np.arange(n))
    T = np.full((n, n), np.inf)
    owners = np.flatnonzero(has)
    for i in owners:
        src = ni[starts[i]:(starts[i + 1] if i + 1 < n else len(ni))]
        d = dijkstra(adj, indices=np.unique(src), min_only=True)
        row = np.minimum.reduceat(d[ni], starts[owners])
        T[i, owners] = row
    np.fill_diagonal(T, 0.0)
    off = ~np.eye(n, dtype=bool)
    fin = np.isfinite(T) & off
    return Travel(T=float(T[fin].mean()) if fin.any() else float("nan"),
                  unreached=int((~np.isfinite(T) & off).sum()), pairs=n * (n - 1),
                  served=served, peel=peel(ctx, served), matrix=T)


def peel(ctx: Ctx, served: np.ndarray) -> float:
    """The deepest BFS layer from the served parcels over the parcel adjacency (1: all served)."""
    if not served.any():
        return float("inf")
    depth = np.full(len(served), np.inf)
    depth[served] = 1
    frontier = list(np.flatnonzero(served))
    k = 1
    while frontier:
        k += 1
        nxt = [j for i in frontier for j in ctx.adj[i] if not np.isfinite(depth[j])]
        nxt = sorted(set(nxt))
        depth[nxt] = k
        frontier = nxt
    return float(depth.max())


# ------------------------------------------------------------------ the bisection stage

class Bisection(NamedTuple):
    lines: np.ndarray            # the new edges, LineStrings
    new_m: float                 # their length
    path_m: float                # the trimmed path's whole length (their bisecting_road's count)
    ratio: float                 # road / straight distance of the chosen pair
    ends: tuple[tuple[float, float], tuple[float, float]]


def bisect_once(ctx: Ctx, net: Net) -> Bisection:
    """One bisection on `net` (module docstring: their code's rule)."""
    g = built(ctx, net.lines, walk=True)
    road_node = np.zeros(len(g.xy), bool)
    road_node[g.eu[g.road]] = True
    road_node[g.ev[g.road]] = True
    cand = np.flatnonzero(road_node & g.corner)
    road_adj = csr(g, g.ew, g.road)
    D = dijkstra(road_adj, indices=cand)[:, cand]
    G = np.linalg.norm(g.xy[cand][:, None] - g.xy[cand][None], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((G > 1e-6) & np.isfinite(D), D / G, -np.inf)
    np.fill_diagonal(r, -np.inf)
    i, j = np.unravel_index(int(np.argmax(r)), r.shape)
    s, t = int(cand[i]), int(cand[j])
    full = csr(g, g.ew, ~g.join)
    _, pred = dijkstra(full, indices=s, return_predecessors=True)
    path = [t]
    while path[-1] != s:
        path.append(int(pred[path[-1]]))
    path.reverse()
    for _ in range(2):                         # shorten_path, from each end
        while len(path) > 2 and road_node[path[1]]:
            path.pop(0)
        path.reverse()
    built_road = g.road & ~g.join
    road_pairs = set(zip(np.minimum(g.eu[built_road], g.ev[built_road]).tolist(),
                         np.maximum(g.eu[built_road], g.ev[built_road]).tolist(), strict=True))
    new = [(a, c) for a, c in zip(path[:-1], path[1:], strict=True)
           if (min(a, c), max(a, c)) not in road_pairs]
    lines = shapely.linestrings(np.stack([g.xy[[a for a, _ in new]], g.xy[[c for _, c in new]]],
                                         axis=1)) if new else np.array([], dtype=object)
    seg = np.linalg.norm(np.diff(g.xy[path], axis=0), axis=1)
    return Bisection(lines=lines, new_m=float(shapely.length(lines).sum()) if new else 0.0,
                     path_m=float(seg.sum()), ratio=float(r[i, j]),
                     ends=(tuple(g.xy[s]), tuple(g.xy[t])))


def with_lines(net: Net, add: np.ndarray, stop: int) -> Net:
    lines = np.concatenate([net.lines, add])
    return Net(method=net.method, stop=stop, lines=lines, paved=paved_of(lines), stage="bisect")


def paved_of(lines: np.ndarray) -> Any:
    if len(lines) == 0:
        return shapely.Polygon()
    return road_corridor(with_width(gpd.GeoDataFrame(geometry=list(lines)), ROAD_W))


# ------------------------------------------------------------------ the networks

def line_net(method: str, stop: int, roads: gpd.GeoDataFrame) -> Net:
    lines = np.array(roads.geometry, dtype=object)
    return Net(method=method, stop=stop, lines=lines,
               paved=road_corridor(roads) if len(roads) else shapely.Polygon(), stage="build")


def topology_kept(site: Site) -> list[Net]:
    roads = gpd.read_parquet(site.kept / "topology_built_roads.parquet")
    ends = roads.groupby("path")["index"].max().to_numpy() + 1     # prefix sizes at path ends
    grid = json.loads((site.kept / f"{TOPOLOGY_GRID}_grid.json").read_text())
    stops = {int(r["stop"]) for r in grid["rows"] if r["status"] == "frame"}
    keep = sorted(set(ends[::4].tolist()) | {int(ends[-1])} | stops)
    return [line_net("topology", int(m), roads.iloc[:m]) for m in keep]


def skeleton_lines(region_local: Any, geo: Any) -> np.ndarray:
    """networks.skeleton's centreline as lines (block CRS), simplified by SKEL_SIMPLIFY."""
    import networks as nw  # noqa: PLC0415

    from reblock.methods.betweenness.thinning import skeletonize  # noqa: PLC0415
    pix = nw.SKEL_PIX
    x0, y0, x1, y1 = region_local.bounds
    lo = np.array([x0, y0]) - 2 * pix
    shape = (int(np.ceil((x1 - lo[0]) / pix)) + 3, int(np.ceil((y1 - lo[1]) / pix)) + 3)
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    shapely.prepare(region_local)
    on = shapely.contains_xy(region_local, lo[0] + (i.ravel() + 0.5) * pix,
                             lo[1] + (j.ravel() + 0.5) * pix).reshape(shape)
    ii, jj = np.nonzero(skeletonize(on))
    keep = set(zip(ii.tolist(), jj.tolist(), strict=True))
    segs = [((a + 0.5, b + 0.5), (a + da + 0.5, b + db + 0.5)) for a, b in keep
            for da, db in ((0, 1), (1, -1), (1, 0), (1, 1)) if (a + da, b + db) in keep]
    xy = lo + pix * np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)
    lines = shapely.get_parts(shapely.line_merge(shapely.multilinestrings(shapely.linestrings(xy))))
    lines = shapely.simplify(lines, SKEL_SIMPLIFY)
    return np.asarray([geo.utm(g) for g in lines])


def joined_to_street(lines: np.ndarray, paved: Any, street: np.ndarray) -> np.ndarray:
    """`lines` and, for each free end inside the corridor that comes within the corridor's reach of
    the street (its distance to the paved edge, plus a raster cell), a straight join to it."""
    st = shapely.union_all(street)
    ends = shapely.get_coordinates(np.concatenate([shapely.get_point(lines, 0),
                                                   shapely.get_point(lines, -1)]))
    pts = shapely.points(ends)
    hit, _ = shapely.STRtree(lines).query(pts, predicate="dwithin", distance=1e-6)
    deg = np.bincount(hit, minlength=len(pts))
    reach = shapely.distance(pts, shapely.boundary(paved)) + 1.5
    near = shapely.distance(pts, st)
    add = [shapely.shortest_line(p, st) for p, d_, r_, n_ in zip(pts, deg, reach, near,
                                                                strict=True)
           if d_ == 1 and n_ <= r_ and n_ > STREET_TOL]
    return np.concatenate([lines, np.asarray(add, dtype=object)]) if add else lines


def shock(b: Any, site: Site) -> list[Net]:
    import networks as nw  # noqa: PLC0415
    grown = nw.Grown(b, nw.gr.OUT / site.bid / SHOCK, targets=())
    street = shapely.get_parts(np.array(b.streets.geometry, dtype=object))
    out = []
    for s in range(1, grown.size + 1):
        reg = grown.region(s)
        paved = grown.geo.utm(reg)
        lines = joined_to_street(skeleton_lines(reg, grown.geo), paved, street)
        out.append(Net(method="shock", stop=s, lines=lines, paved=paved, stage="build"))
    return out


def cycle(site: Site) -> list[Net]:
    roads = gpd.read_parquet(RUNS / "cycle_native.ordered.parquet")
    return [line_net("cycle_native", m, roads.iloc[:m]) for m in range(1, len(roads) + 1)]


def whole(method: str, path: Path) -> Net:
    roads = gpd.read_parquet(path)
    return line_net(method, len(roads), roads)


def samples() -> list[Net]:
    return [whole(p.stem, p) for p in sorted(RUNS.glob("topology.a*.s*.parquet"))]


# ------------------------------------------------------------------ rows

def row(ctx: Ctx, net: Net, extra: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    graphs = {False: built(ctx, net.lines, walk=False), True: built(ctx, net.lines, walk=True)}
    for conv in CONVENTIONS:
        tr = travel(ctx, net, conv, g=graphs[conv.walk])
        out.append(dict(method=net.method, stage=net.stage, stop=net.stop, convention=conv.name,
                        new_m=float(shapely.length(net.lines).sum()) if len(net.lines) else 0.0,
                        T=tr.T, unreached=tr.unreached, pairs=tr.pairs,
                        served=float(tr.served.mean()), peel=tr.peel, **extra))
    return out


def bisected(ctx: Ctx, net: Net, log: list[dict[str, Any]]) -> list[Net]:
    """`net` and its BISECTIONS bisections, each logged."""
    nets, cur = [net], net
    for k in range(1, BISECTIONS + 1):
        t0 = time.time()
        bi = bisect_once(ctx, cur)
        cur = with_lines(cur, bi.lines, stop=k)
        nets.append(cur)
        log.append(dict(method=net.method, k=k, new_m=bi.new_m, path_m=bi.path_m,
                        ratio=bi.ratio, ends=bi.ends, s=time.time() - t0,
                        wkt=[shapely.to_wkt(g_, rounding_precision=4) for g_ in bi.lines]))
        print(f"  {net.method} bisection {k}: {bi.new_m:.1f} m new ({bi.path_m:.1f} m path), "
              f"ratio {bi.ratio:.2f}", flush=True)
    return nets


def kept_whole(site: Site) -> Net:
    return topology_kept(site)[-1]


def part_nets(part: str, ctx: Ctx, site: Site, log: list[dict[str, Any]]) -> list[Net]:
    """The states a part scores (PARTS)."""
    match part:
        case "none":
            return [Net("none", 0, np.array([], dtype=object), shapely.Polygon(), "build")]
        case "topology":
            kept = topology_kept(site)
            return kept + bisected(ctx, kept[-1], log)[1:]
        case "steiner":
            return bisected(ctx, whole("steiner", RUNS / "steiner.ordered.parquet"), log)
        case "cycle_native":
            return cycle(site)
        case "shock":
            return shock(ctx.b, site)
        case "samples":
            smp = samples()
            best = min(smp, key=lambda n: float(shapely.length(n.lines).sum()))
            kept_m = float(shapely.length(kept_whole(site).lines).sum())
            more = bisected(ctx, best, log)[1:] if shapely.length(best.lines).sum() < kept_m else []
            return smp + more
    raise SystemExit(f"{part}: not one of {PARTS}")


PARTS = ("none", "topology", "steiner", "cycle_native", "shock", "samples")


def rows(bid: str, part: str) -> None:
    """One part's rows to rows/<part>.parquet, its bisections to rows/<part>.bisections.json."""
    site = Site.of(bid)
    [b] = common.build_blocks([bid])
    ctx = ctx_of(b)
    ROWS.mkdir(exist_ok=True)
    log: list[dict[str, Any]] = []
    t0 = time.time()
    nets = part_nets(part, ctx, site, log)
    out = [r for net in nets for r in row(ctx, net, {})]
    pd.DataFrame(out).to_parquet(ROWS / f"{part}.parquet")
    if log:
        (ROWS / f"{part}.bisections.json").write_text(json.dumps(log, indent=1) + "\n")
    print(f"{part}: {len(nets)} states ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ our lenses at the same points

def lenses(bid: str, which: str) -> None:
    """D and L_P of cycle_native's prefixes at DEMO_D and whole, and the floor's whole network
    (and its and topology's bisected end states), scored as topology_run.py scores a network."""
    import networks as nw  # noqa: PLC0415
    import road_simp as rs  # noqa: PLC0415
    [b] = common.build_blocks([bid])
    w = nw.area_weights(b)
    metric = nw.cp.gcl.complex_of(nw.metrics.RUNTIME, factorization=nw.cp.gap_factor.SuperLU())
    none = np.zeros(len(b.buildings.outlines), dtype=bool)

    def score(roads: gpd.GeoDataFrame) -> dict[str, float]:
        ground = shapely.intersection(road_corridor(roads), b.boundary)
        _m, _l, r = rs.scored(b, metric, ground, cleared=none)
        return dict(D=float(r["D_exact"]), L_P=float(r["L_P"]),
                    D_measure=common.displacement(b, roads, w, measure=common.MEASURE))

    path = HERE / "lenses.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    if which == "trees":
        # the floor, the best sample, and each bisection stage's end state (kept, best, floor)
        site = Site.of(bid)
        bases = {"steiner": gpd.read_parquet(RUNS / "steiner.ordered.parquet"),
                 "topology": gpd.read_parquet(site.kept / "topology_roads.parquet")}
        for p_ in RUNS.glob("topology.a*.s*.parquet"):
            bases[p_.stem] = gpd.read_parquet(p_)
        bis: dict[str, list[str]] = {}
        for p_ in ROWS.glob("*.bisections.json"):
            for r_ in json.loads(p_.read_text()):
                bis.setdefault(r_["method"], []).extend(r_["wkt"])
        for name, roads in bases.items():
            if name != "topology":
                out[f"{name}@whole"] = dict(stop=len(roads), **score(roads))
                print(name, out[f"{name}@whole"], flush=True)
            if name in bis:
                more = with_width(gpd.GeoDataFrame(geometry=shapely.from_wkt(bis[name]),
                                                   crs=roads.crs), ROAD_W)
                both = pd.concat([roads[["geometry", "width_m"]], more], ignore_index=True)
                out[f"{name}@bisected{BISECTIONS}"] = dict(stop=len(both), **score(
                    gpd.GeoDataFrame(both, geometry="geometry", crs=roads.crs)))
                print(name, "bisected", out[f"{name}@bisected{BISECTIONS}"], flush=True)
        path.write_text(json.dumps(out, indent=1) + "\n")
        return
    cn = gpd.read_parquet(RUNS / "cycle_native.ordered.parquet")
    Ds = [common.displacement(b, cn.iloc[:m], w, measure=common.MEASURE)
          for m in range(1, len(cn) + 1)]
    out["cycle_native_D_measure"] = Ds
    for t in DEMO_D:
        m = next((k + 1 for k, d in enumerate(Ds) if d >= t), len(cn))
        out[f"cycle_native@{t:g}"] = dict(stop=m, **score(cn.iloc[:m]))
        print(t, out[f"cycle_native@{t:g}"], flush=True)
    out["cycle_native@whole"] = dict(stop=len(cn), **score(cn))
    path.write_text(json.dumps(out, indent=1) + "\n")


# ------------------------------------------------------------------ the cross-check

def cross(bid: str) -> None:
    """`travel` under THEIRS against crosscheck.py's matrix (their shortest_path_p2p_matrix) on
    the kept topology network: faces matched to parcels by nearest centroid, then the means and
    the per-pair differences, to crosscheck.json."""
    site = Site.of(bid)
    [b] = common.build_blocks([bid])
    ctx = ctx_of(b)
    z = np.load(RUNS / "crosscheck_topology.npz")
    theirs = z["T"]
    face = shapely.points(z["centroid"])
    pc = shapely.centroid(ctx.parcels)
    _, j = shapely.STRtree(pc).query_nearest(face, return_distance=False)
    if len(set(j.tolist())) != len(j) or len(j) != len(ctx.parcels):
        raise AssertionError(f"{len(j)} faces onto {len(set(j.tolist()))} of "
                             f"{len(ctx.parcels)} parcels")
    tr = travel(ctx, kept_whole(site), THEIRS)
    mine = tr.matrix[np.ix_(j, j)]
    off = ~np.eye(len(j), dtype=bool)
    both = off & np.isfinite(mine) & np.isfinite(theirs)
    rel = (mine[both] - theirs[both]) / np.maximum(theirs[both], 1.0)
    no_join = travel_without_joins(ctx, kept_whole(site))
    out = dict(theirs_mean=float(z["mean"]), theirs_mean_finite=float(theirs[both].mean()),
               theirs_unreached=int((off & ~np.isfinite(theirs)).sum()),
               theirs_s=float(z["seconds"]),
               mine_T=tr.T, mine_unreached=tr.unreached, mine_over_theirs=tr.T / float(z["mean"]),
               pair_rel_median=float(np.median(rel)), pair_rel_p05=float(np.percentile(rel, 5)),
               pair_rel_p95=float(np.percentile(rel, 95)),
               pair_abs_median_m=float(np.median(np.abs(mine[both] - theirs[both]))),
               pairs_within_1pct=float((np.abs(rel) <= 0.01).mean()),
               corr=float(np.corrcoef(mine[both], theirs[both])[0, 1]),
               mine_T_no_culdesac=no_join)
    (HERE / "crosscheck.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


def travel_without_joins(ctx: Ctx, net: Net) -> float:
    """THEIRS' T with no cul-de-sac joins (the convention's sensitivity)."""
    g = built(ctx, net.lines, walk=False)
    g = g._replace(road=g.road & ~g.join)
    return travel(ctx, net, THEIRS, g=g).T


# ------------------------------------------------------------------ the report

COLOURS = {"topology": "#2a78d6", "topology best": "#eb6834", "steiner": "#1baf7a",
           "shock": "#eda100", "cycle_native": "#e87ba4"}
NAMES = {"topology": "topology (kept run, a10 s0)", "topology best": "topology (best sample)",
         "steiner": "group Steiner floor", "shock": "Shock Reblock", "cycle_native": "Loop Network"}


def frames() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted(ROWS.glob("*.parquet"))],
                     ignore_index=True)


def demo_stops(site: Site) -> dict[str, dict[float, dict[str, Any]]]:
    """Per method its stop, D and L_P at each DEMO_D (and `whole`), from the grid tables and
    lenses.json."""
    out: dict[str, dict[Any, dict[str, Any]]] = {}
    for m, key in (("topology", TOPOLOGY_GRID), ("shock", SHOCK)):
        g = json.loads((site.kept / f"{key}_grid.json").read_text())
        by = {r["budget"]: r for r in g["rows"]}
        out[m] = {t: dict(stop=by[t]["stop"], D=by[t]["D"], L_P=by[t]["L_P"]) for t in DEMO_D
                  if by[t]["status"] != "stopped"}
        out[m]["whole"] = dict(stop=g["whole"]["stop"], D=g["whole"]["D"], L_P=g["whole"]["L_P"])
    lj = json.loads((HERE / "lenses.json").read_text())
    out["cycle_native"] = {t: lj[f"cycle_native@{t:g}"] for t in DEMO_D}
    out["cycle_native"]["whole"] = lj["cycle_native@whole"]
    out["steiner"] = {"whole": lj["steiner@whole"]}
    out["lenses_json"] = lj
    return out


def report(bid: str) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    site = Site.of(bid)
    df = frames()
    smp = df[df.method.str.startswith("topology.a")]
    best = None
    if len(smp):
        lens_ = smp[(smp.stage == "build") & (smp.convention == "line")].set_index("method").new_m
        best = str(lens_.idxmin())
        print("topology samples, new road m:", lens_.round(1).to_dict(), flush=True)
    stops = demo_stops(site)
    summary: dict[str, Any] = {}
    # objective 1 across topology's runs (each run's log line; the kept run's record) and the floor
    runs_m = {"a10 s0 (kept)": json.loads((site.kept / "topology.json").read_text())["road_m"]}
    for p_ in sorted(RUNS.glob("topology.a*.s*.log")):
        rec = json.loads(next(ln for ln in p_.read_text().splitlines() if ln.startswith("{")))
        runs_m[f"a{rec['alpha']:g} s{rec['seed']}"] = rec["road_m"]
    floor = json.loads((RUNS / "steiner.json").read_text())
    lb = floor["lb"]                      # quota.solve's: the LP's or the MILP's, the larger
    summary["objective1"] = dict(runs=runs_m, best=min(runs_m.values()),
                                 floor_tree=floor["tree_m"], floor_lower_bound=lb,
                                 floor_proven=floor["proven"])
    print("objective 1, new road to universal access (m):")
    for k_, v_ in runs_m.items():
        print(f"  topology {k_:<14}{v_:8.1f}  {v_ / floor['tree_m'] - 1:+.1%} over the floor tree, "
              f"{v_ / lb - 1:+.1%} over its lower bound")
    print(f"  group Steiner tree  {floor['tree_m']:8.1f}  (lower bound {lb:.1f}, gap "
          f"{floor['tree_m'] / lb - 1:.1%}{', proven optimal' if floor['proven'] else ''})")
    print(f"\n{'method':<28}{'conv':<10}{'first universal':>18}{'new m':>9}{'T':>8}  "
          "final: served / peel / new m")
    curves: dict[str, pd.DataFrame] = {}
    for m in ("topology", "topology best", "steiner", "shock", "cycle_native"):
        src = best if m == "topology best" else m
        if src is None:
            continue
        d = df[df.method == src]
        curves[m] = d
        for conv in ("line", "corridor"):
            c = d[(d.convention == conv) & (d.stage == "build")].sort_values("new_m")
            uni = c[c.served >= 1.0]
            fin = c.iloc[-1]
            first = uni.iloc[0] if len(uni) else None
            summary[f"{m}|{conv}"] = dict(
                universal=first is not None,
                first_stop=None if first is None else int(first.stop),
                first_new_m=None if first is None else float(first.new_m),
                first_T=None if first is None else float(first["T"]),
                final_served=float(fin.served), final_peel=float(fin.peel),
                final_new_m=float(fin.new_m), final_T=float(fin["T"]),
                final_unreached=int(fin.unreached))
            fs = "-" if first is None else f"stop {int(first.stop)}"
            fm = "-" if first is None else f"{first.new_m:.0f}"
            ft = "-" if first is None else f"{first['T']:.1f}"
            print(f"{NAMES[m]:<28}{conv:<10}{fs:>18}{fm:>9}{ft:>8}  "
                  f"{fin.served:.3f} / {fin.peel:g} / {fin.new_m:.0f}")
        for t, st in stops.get(m if m != "topology best" else "", {}).items():
            for conv in ("line", "corridor", "line+walk"):
                r = d[(d.convention == conv) & (d.stage == "build") & (d.stop == st["stop"])]
                if len(r) == 0:
                    continue
                r0 = r.iloc[0]
                summary[f"{m}|{conv}|D{t}"] = dict(stop=int(st["stop"]), D=st["D"], L_P=st["L_P"],
                                                    new_m=float(r0.new_m), served=float(r0.served),
                                                    peel=float(r0.peel), T=float(r0["T"]),
                                                    unreached=int(r0.unreached))
    bis = {}
    for p in sorted(ROWS.glob("*.bisections.json")):
        for r in json.loads(p.read_text()):
            bis.setdefault(r["method"], []).append(r)
    summary["bisections"] = bis
    # matched road length: T on each road-only curve (where every pair is reachable) interpolated
    def curve(m: str, conv: str) -> pd.DataFrame:
        d = curves[m]
        c = d[(d.convention == conv) & (d.unreached == 0)].sort_values("new_m")
        return c[["new_m", "T", "stage", "stop"]]

    # the lengths compared at: every network's universal-access and whole lengths, and the ends
    # of each bisection stage
    marks = sorted({float(x) for m in curves for x in (
        curves[m][curves[m].convention == "line"].groupby("stage").new_m.max().tolist()
        + [summary[f"{m}|line"]["first_new_m"] or np.nan]) if np.isfinite(x)})
    matched: dict[str, dict[str, float | None]] = {}
    for conv in ("line", "corridor", "line+walk"):
        for m in curves:
            c = curve(m, conv)
            # the lower envelope: T at length L is the best state built with at most L
            env = np.minimum.accumulate(c["T"].to_numpy()) if len(c) else np.array([])
            xs = c["new_m"].to_numpy()
            matched[f"{m}|{conv}"] = {
                f"{L:.1f}": (float(env[np.searchsorted(xs, L, side="right") - 1])
                             if len(xs) and L >= xs[0] else None) for L in marks}
    summary["matched_T_best_at_or_below_L"] = matched
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps(matched, indent=1))
    # the curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for ax, conv, title in ((axes[0], "line", "Their T: roads only, parcels touching a centreline"),
                            (axes[1], "line+walk",
                             f"Roads, plus parcel boundaries at {WALK_X:g}x (every prefix)")):
        for m, d in curves.items():
            c = d[d.convention == conv].sort_values(["stage", "new_m"])
            if conv == "line":
                c = c[c.unreached == 0]
            if len(c) == 0:
                continue
            b_ = c[c.stage == "build"]
            s_ = c[c.stage == "bisect"]
            col = COLOURS[m]
            if len(b_) > 1:
                ax.plot(b_.new_m, b_["T"], color=col, lw=2, label=NAMES[m])
            elif len(b_) == 1:
                ax.plot(b_.new_m, b_["T"], "o", color=col, ms=8, label=NAMES[m])
            if len(s_):
                xs = np.r_[b_.new_m.iloc[-1:], s_.new_m] if len(b_) else s_.new_m
                ys = np.r_[b_["T"].iloc[-1:], s_["T"]] if len(b_) else s_["T"]
                ax.plot(xs, ys, "--", color=col, lw=2, marker="o", ms=5)
            u = summary.get(f"{m}|line", {})
            if u.get("first_new_m") is not None and conv == "line+walk":
                r = c[(c.stage == "build") & (c.stop == u["first_stop"])]
                if len(r):
                    ax.plot(r.new_m, r["T"], "*", color=col, ms=14, mec="#0b0b0b", mew=0.6)
        if conv == "line":
            never = [NAMES[m] for m in curves if summary[f"{m}|line"]["first_new_m"] is None]
            if never:
                ax.text(0.02, 0.02, "Never universally accessible, so T undefined:\n"
                        + "\n".join(never), transform=ax.transAxes, fontsize=8,
                        color="#52514e", va="bottom")
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.set_xlabel("new road length (m; the block's street, 712 m, not counted)",
                      color="#52514e")
        ax.set_ylabel("mean parcel-to-parcel travel distance T (m)", color="#52514e")
        ax.grid(color="#e6e5e0", lw=0.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.legend(frameon=False, fontsize=8,
                  loc="upper center" if conv == "line" else "upper right")
    fig.suptitle(f"{bid}: topology's objectives (dashed: their bisection stage; star: universal "
                 "access)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "curves.png", dpi=130)
    print(f"{HERE / 'curves.png'}", flush=True)


if __name__ == "__main__":
    rr._check_single_threaded()
    match sys.argv[1:]:
        case ["rows", bid, part]:
            rows(bid, part)
        case ["lenses", bid, ("cycle_native" | "trees") as which]:
            lenses(bid, which)
        case ["cross", bid]:
            cross(bid)
        case ["report", bid]:
            report(bid)
        case _:
            raise SystemExit(__doc__)
