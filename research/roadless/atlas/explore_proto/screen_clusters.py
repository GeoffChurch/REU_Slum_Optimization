"""The screen's anomalous clusters of whole blocks (owner 2026-11-03), for the Demo Lineup
artifact's height map: two trees of candidate clusters, each scanned for clusters scoring
anomalously high.

The blocks (the site's ScreenMap bundle) each carry a weight, their buildings n, and a value, their
log10 score (depth_density_proxy sqrt(nA)/P n/A, or density_compactness n/P^2; a score of 0 at
the city's 1st percentile).

Trees (a Strategy each: Tree):
  street: the component tree of the scores on the blocks' street adjacency. The bundle's blocks
    are faces of the street network, so neighbours share a street centreline: two blocks touch
    where their boundaries come within TOUCH_M (the bundle's 5 m simplification) along at least
    SHARED_M (not at a corner). Blocks are added in descending score, each joining the
    components of its added neighbours (union-find); each addition is a node, the component it
    makes. A greedy flood from any block (the best-scoring neighbour each step) passes through
    exactly these components, and floods from two blocks first coincide where the tree merges
    them. Milliseconds a tree.
  hull: agglomerative, with no adjacency rule: every block a cluster, then repeatedly the pair of
    neighbouring clusters (neighbours in the Delaunay triangulation of the blocks' centroids,
    only a candidate set) whose union scores best by H = its mean log score + log10 of its fill
    (its blocks' area over its convex hull's): a merge across empty ground dilutes the hull, so
    compactness comes from the objective. The depth term of the screen cannot score a hull (its
    streets run inside it, and at uniform density sqrt(nA)/P grows as sqrt(A)), so H is
    intensive. Each merge a node. Greedy: a merge can lower H.

Scans, on every node:
  scan (both trees): the Gaussian log-likelihood ratio for "this cluster's mean log score is
    above the rest of the city's", buildings as weights (Patil and Taillie's upper-level-set scan
    statistic in the street tree): W_Z W_O / W (m_Z - m_O)^2 / (2 sigma^2) where m_Z > m_O and
    the cluster holds at most MAX_SHARE of the city's buildings (Kulldorff's default). Its p the
    rank of the node's ratio among the trees' largest ratios over shuffles of the blocks' log
    scores among the blocks (their weights and geometry kept), the tree rebuilt each time.
  local (street tree): persistence, how far a cluster stands above the lowest block one must
    cross to reach higher ground: when two components merge at block v, the one with the lower
    peak dies, its persistence its peak's log score less v's, the cluster the component as it was
    just before. Its p the rank among the trees' largest persistences over the same shuffles.
Per scan the TOP clusters by statistic, disjoint (no two share a block). Beside each, its
contrast: its mean log score less the mean of the blocks touching it (street adjacency) not in it.

    S=research/roadless/atlas/explore_proto/screen_clusters.py  # from the repo root, PYTHONPATH=.
    uv run python $S ~/src/wt_demo <out dir>      # clusters_<city>.json per city, both metrics
"""
from __future__ import annotations

import heapq
import json
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Protocol

import numpy as np
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy.spatial import Delaunay

TOUCH_M = 0.5          # m: street adjacency: boundaries this close (the bundle's simplification)
SHARED_M = 5.0         # m: ... along at least this much of a boundary (not a corner)
MAX_SHARE = 0.5        # a scanned cluster's buildings at most this share of the city's
TOP = 8                # clusters per scan
NPERM_STREET = 999     # shuffles for the street tree's p (milliseconds a tree)
NPERM_HULL = 47        # shuffles for the hull tree's p (a re-agglomeration each)
SEED = 0
WORKERS = 16
CITIES = ("capetown", "nairobi")
METRICS = {"depth_density_proxy": lambda n, A, P: np.sqrt(n * A) / P * n / A,
           "density_compactness": lambda n, A, P: n / P ** 2}

I64 = NDArray[np.int64]
F64 = NDArray[np.float64]


# ---------------------------------------------------------------- the blocks

@dataclass(frozen=True)
class Blocks:
    polys: list
    w: F64                   # buildings
    area: F64                # m^2
    nbr_ptr: I64             # street adjacency, CSR
    nbr: I64
    cen: F64                 # centroids
    hull_pts: list           # per block its convex hull's vertices (k x 2)


def blocks_of(city: dict) -> Blocks:
    polys = [shapely.Polygon(r[0], r[1:]) for r in city["rings"]]
    t = shapely.STRtree(polys)
    a, b = t.query(polys, predicate="dwithin", distance=TOUCH_M)
    keep = a < b
    a, b = a[keep], b[keep]
    bnd = shapely.boundary(np.asarray(polys, dtype=object))
    shared = shapely.length(shapely.intersection(bnd[a], shapely.buffer(bnd[b], TOUCH_M)))
    a, b = a[shared >= SHARED_M], b[shared >= SHARED_M]
    n = len(polys)
    src, dst = np.concatenate([a, b]), np.concatenate([b, a])
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ptr, src + 1, 1)
    hulls = [np.asarray(shapely.convex_hull(p).exterior.coords)[:-1] for p in polys]
    c = shapely.centroid(np.asarray(polys, dtype=object))
    return Blocks(polys=polys, w=np.asarray(city["n"], dtype=float),
                  area=np.asarray(city["area_m2"], dtype=float), nbr_ptr=np.cumsum(ptr),
                  nbr=dst[order].astype(np.int64),
                  cen=np.column_stack([shapely.get_x(c), shapely.get_y(c)]), hull_pts=hulls)


def log_scores(city: dict, metric: str) -> F64:
    n, A, P = (np.asarray(city[k], dtype=float) for k in ("n", "area_m2", "perimeter_m"))
    s = METRICS[metric](n, A, P)
    v = np.log10(np.where(s > 0, s, np.nan))
    return np.where(np.isfinite(v), v, np.nanpercentile(v, 1))


@njit(cache=True)
def llr(Wz: float, Sz: float, W: float, S: float, var: float, cap: float) -> float:
    if Wz > cap or Wz >= W:
        return 0.0
    mz, mo = Sz / Wz, (S - Sz) / (W - Wz)
    if mz <= mo:
        return 0.0
    return Wz * (W - Wz) / W * (mz - mo) ** 2 / (2.0 * var)


# ---------------------------------------------------------------- the street tree

@njit(cache=True)
def _find(par: I64, x: int) -> int:
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


@njit(cache=True)
def street_tree(v: F64, w: F64, ptr: I64, nbr: I64, cap_share: float
                ) -> tuple[I64, F64, F64, F64, F64]:
    """Per node (the addition of block i, its component then) its parent node (-1 the root of
    its piece of the city), its ratio, its persistence (nan unless it dies at a merge) and its
    W and S."""
    n = len(v)
    order = np.argsort(-v)
    par = np.arange(n)
    added = np.zeros(n, dtype=np.bool_)
    Wc, Sc, peak = np.zeros(n), np.zeros(n), np.zeros(n)
    cur = np.arange(n)                       # per root: the node of its latest addition
    parent = np.full(n, -1, dtype=np.int64)
    ratio, pers = np.zeros(n), np.full(n, np.nan)
    Wn, Sn = np.zeros(n), np.zeros(n)
    W, S = w.sum(), (w * v).sum()
    m = S / W
    var = (w * (v - m) ** 2).sum() / W
    cap = cap_share * W
    for i in order:
        added[i] = True
        roots = np.empty(ptr[i + 1] - ptr[i], dtype=np.int64)
        cnt = 0
        for k in range(ptr[i], ptr[i + 1]):
            j = nbr[k]
            if not added[j]:
                continue
            r = _find(par, j)
            seen = False
            for q in range(cnt):
                if roots[q] == r:
                    seen = True
            if not seen:
                roots[cnt] = r
                cnt += 1
        Wc[i], Sc[i], peak[i] = w[i], w[i] * v[i], v[i]
        top = i
        if cnt:
            # the highest peak among the components i joins survives; every other dies here, as
            # it was just before (its latest node)
            top = roots[0]
            for q in range(1, cnt):
                if peak[roots[q]] > peak[top]:
                    top = roots[q]
            for q in range(cnt):
                r = roots[q]
                parent[cur[r]] = i
                if r != top:
                    pers[cur[r]] = peak[r] - v[i]
                    par[r] = top
                    Wc[top] += Wc[r]
                    Sc[top] += Sc[r]
            par[i] = top
            Wc[top] += w[i]
            Sc[top] += w[i] * v[i]
        cur[top] = i
        Wn[i], Sn[i] = Wc[top], Sc[top]
        ratio[i] = llr(Wc[top], Sc[top], W, S, var, cap)
    return parent, ratio, pers, Wn, Sn


# ---------------------------------------------------------------- the hull tree

@njit(cache=True)
def _cross(o0: float, o1: float, a0: float, a1: float, b0: float, b1: float) -> float:
    return (a0 - o0) * (b1 - o1) - (a1 - o1) * (b0 - o0)


@njit(cache=True)
def hull(p: F64) -> F64:
    """The convex hull's vertices of the points p (Andrew's monotone chain)."""
    idx = np.argsort(p[:, 1], kind="mergesort")
    idx = idx[np.argsort(p[idx, 0], kind="mergesort")]          # by x, then y
    q = p[idx]
    n = len(q)
    h = np.empty((2 * n, 2))
    k = 0
    for i in range(n):
        while k >= 2 and _cross(h[k - 2, 0], h[k - 2, 1], h[k - 1, 0], h[k - 1, 1],
                                q[i, 0], q[i, 1]) <= 0:
            k -= 1
        h[k] = q[i]
        k += 1
    t = k + 1
    for i in range(n - 2, -1, -1):
        while k >= t and _cross(h[k - 2, 0], h[k - 2, 1], h[k - 1, 0], h[k - 1, 1],
                                q[i, 0], q[i, 1]) <= 0:
            k -= 1
        h[k] = q[i]
        k += 1
    return h[:k - 1].copy()


@njit(cache=True)
def area_of(h: F64) -> float:
    a = 0.0
    for i in range(len(h)):
        j = (i + 1) % len(h)
        a += h[i, 0] * h[j, 1] - h[j, 0] * h[i, 1]
    return abs(a) / 2.0


def hull_tree(v: F64, b: Blocks, edges: I64, cap_share: float) -> tuple[I64, F64, F64, F64, I64]:
    """The agglomeration's nodes: blocks 0..n-1, then each merge n, n+1, ...; per node its
    parent, its ratio, its W and S, and its first two children (-1 for a block)."""
    n = len(v)
    W, S = b.w.sum(), (b.w * v).sum()
    var = (b.w * (v - S / W) ** 2).sum() / W
    cap = cap_share * W
    pts = list(b.hull_pts)
    Wn, Sn, An = list(b.w), list(b.w * v), list(b.area)
    nb: list[set[int]] = [set() for _ in range(n)]
    for x, y in edges:
        nb[x].add(int(y))
        nb[y].add(int(x))
    alive = [True] * n
    parent = [-1] * n
    kids = [(-1, -1)] * n

    def H(x: int, y: int) -> tuple[float, F64]:
        h = hull(np.vstack([pts[x], pts[y]]))
        fill = (An[x] + An[y]) / max(area_of(h), 1e-9)
        return (Sn[x] + Sn[y]) / (Wn[x] + Wn[y]) + np.log10(min(fill, 1.0)), h

    heap = []
    for x in range(n):
        for y in nb[x]:
            if x < y:
                heapq.heappush(heap, (-H(x, y)[0], x, y))
    while heap:
        _, x, y = heapq.heappop(heap)
        if not (alive[x] and alive[y]):
            continue
        _, h = H(x, y)
        z = len(pts)
        pts.append(h)
        Wn.append(Wn[x] + Wn[y])
        Sn.append(Sn[x] + Sn[y])
        An.append(An[x] + An[y])
        alive += [True]
        alive[x] = alive[y] = False
        parent += [-1]
        parent[x] = parent[y] = z
        kids.append((x, y))
        nz = (nb[x] | nb[y]) - {x, y}
        nb.append(nz)
        for u in nz:
            nb[u].discard(x)
            nb[u].discard(y)
            nb[u].add(z)
            heapq.heappush(heap, (-H(z, u)[0], min(z, u), max(z, u)))
    Wa, Sa = np.asarray(Wn), np.asarray(Sn)
    ratio = np.array([llr(Wa[i], Sa[i], W, S, var, cap) for i in range(len(Wa))])
    return np.asarray(parent), ratio, Wa, Sa, np.asarray(kids)


# ---------------------------------------------------------------- trees as a Strategy

class Tree(Protocol):
    name: str
    nperm: int

    def build(self, v: F64) -> dict: ...

    def members(self, t: dict, node: int) -> I64: ...


@dataclass(frozen=True)
class Street:
    b: Blocks
    name: str = "street"
    nperm: int = NPERM_STREET

    def build(self, v: F64) -> dict:
        parent, ratio, pers, Wn, Sn = street_tree(v, self.b.w, self.b.nbr_ptr, self.b.nbr,
                                                  MAX_SHARE)
        return dict(parent=parent, scan=ratio, local=pers, W=Wn, S=Sn)

    def members(self, t: dict, node: int) -> I64:
        kids: dict[int, list[int]] = {}
        for c, p in enumerate(t["parent"]):
            if p >= 0:
                kids.setdefault(int(p), []).append(c)
        out, stack = [], [node]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(kids.get(x, []))
        return np.asarray(out)


@dataclass(frozen=True)
class Hull:
    b: Blocks
    edges: I64
    name: str = "hull"
    nperm: int = NPERM_HULL

    def build(self, v: F64) -> dict:
        parent, ratio, Wn, Sn, kids = hull_tree(v, self.b, self.edges, MAX_SHARE)
        return dict(parent=parent, scan=ratio, W=Wn, S=Sn, kids=kids)

    def members(self, t: dict, node: int) -> I64:
        n = len(self.b.w)
        out, stack = [], [node]
        while stack:
            x = stack.pop()
            if x < n:
                out.append(x)
            else:
                stack.extend(int(c) for c in t["kids"][x])
        return np.asarray(out)


def delaunay_edges(cen: F64) -> I64:
    tri = Delaunay(cen).simplices
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    return np.unique(np.sort(e, axis=1), axis=0)


_JOB: dict = {}


def _null(seed: int) -> dict[str, float]:
    tree, v = _JOB["tree"], _JOB["v"]
    t = tree.build(np.random.default_rng(seed).permutation(v))
    return {k: float(np.nanmax(t[k])) for k in ("scan", "local") if k in t}


def scanned(tree: Tree, v: F64, b: Blocks, floor: float) -> dict:
    t = tree.build(v)
    _JOB.update(tree=tree, v=v)
    seeds = [int(x) for x in np.random.SeedSequence(SEED).generate_state(tree.nperm)]
    with get_context("fork").Pool(WORKERS) as pool:
        null = pool.map(_null, seeds)
    out: dict = {"nperm": tree.nperm, "seed": SEED}
    for kind in ("scan", "local"):
        if kind not in t:
            continue
        stat = np.nan_to_num(t[kind], nan=-np.inf)
        nmax = np.array([x[kind] for x in null])
        used = np.zeros(len(v), dtype=bool)
        found = []
        for node in np.argsort(-stat):
            if not np.isfinite(stat[node]) or stat[node] <= 0 or len(found) == TOP:
                break
            mem = tree.members(t, int(node))
            if used[mem].any():
                continue
            used[mem] = True
            Wz, Sz = float(b.w[mem].sum()), float((b.w[mem] * v[mem]).sum())
            ring = np.setdiff1d(np.unique(np.concatenate(
                [b.nbr[b.nbr_ptr[i]:b.nbr_ptr[i + 1]] for i in mem])), mem)
            contrast = (Sz / Wz - float((b.w[ring] * v[ring]).sum() / b.w[ring].sum())
                        if len(ring) else None)
            found.append(dict(
                members=mem.tolist(), stat=float(stat[node]),
                p=float((1 + (nmax >= stat[node]).sum()) / (1 + len(nmax))),
                buildings=Wz, blocks=int(len(mem)), area_km2=float(b.area[mem].sum() / 1e6),
                mean_score=float(10 ** (Sz / Wz)), x_floor=float(10 ** (Sz / Wz) / floor),
                contrast=contrast,
                cx=float((b.cen[mem, 0] * b.w[mem]).sum() / Wz),
                cy=float((b.cen[mem, 1] * b.w[mem]).sum() / Wz)))
        out[kind] = found
    return out


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root, out = Path(argv[0]).expanduser(), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in CITIES:
        t0 = time.time()
        city = json.loads((root / "examples" / "screen-map" / f"{name}.json").read_text())
        b = blocks_of(city)
        edges = delaunay_edges(b.cen)
        print(f"{name}: {len(b.w)} blocks, {len(b.nbr) // 2} street pairs, {len(edges)} "
              f"Delaunay pairs ({time.time() - t0:.0f}s)", flush=True)
        res: dict = {"touch_m": TOUCH_M, "shared_m": SHARED_M, "max_share": MAX_SHARE}
        for metric in METRICS:
            v = log_scores(city, metric)
            floor = next(f["value"] for f in city["floors"] if f["metric"] == metric)
            res[metric] = {}
            trees: list[Tree] = [Street(b), Hull(b, edges)]
            for tree in trees:
                t1 = time.time()
                r = scanned(tree, v, b, floor)
                res[metric][tree.name] = r
                for kind in ("scan", "local"):
                    if kind in r:
                        top = [(round(c["stat"], 1), round(c["p"], 3), c["blocks"],
                                round(c["x_floor"], 2)) for c in r[kind][:3]]
                        print(f"  {metric} {tree.name} {kind}: {top} ({time.time() - t1:.0f}s)",
                              flush=True)
        (out / f"clusters_{name}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1:])
