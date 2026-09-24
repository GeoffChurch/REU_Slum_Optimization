"""The (cell, direction-bucket) state graph and its two count passes.

Field B (permeability) with the path field's smoothness: one best route per origin-destination
pair, where a route pays for its turning. The state is (cell, direction bucket).

* 16 lattice step directions (king + knight moves). Bucket b is the angular interval between
  directions b and b+1; inside a bucket a path alternates freely between the two bounding steps,
  so a straight road at ANY angle costs only its length.
* A step may also move to an adjacent bucket, paying lam * dmid^2 / L0 (dmid = the difference of
  the two buckets' mid-angles, L0 = 2 m): elastica at the 2 m scale. There is no cooldown, so
  below a radius of ~L0/dmid (~5 m) the charge is for total turning, not for curvature.
* Step cost = step length * mean cost density over the cells it touches (both ends, and for a
  knight move the two cells it crosses); cost density c = 1 + (r0 / max(clearance, res/2))^2 on
  free cells, 50 in buildings.

Egress: every street-band cell is a root in every bucket; per cell, how many homes' best routes
    to the street pass THROUGH it.
All-pairs: one Dijkstra per source home (entered at cost 0 in every bucket); per cell, how many
    (source, target) pairs' best routes pass THROUGH it. A target's route is the pred chain of its
    FIRST settled state (its best bucket); a cell is credited at most once per route, and the
    route's own endpoints are never credited.
"""
from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reblock._jit import njit

BUILDING_COST = 50.0
L0_M = 2.0
NB = 16
# (dx = column, dy = row), counter-clockwise from +x.
DIRS = np.array([(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
                 (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)],
                dtype=np.int64)
THETA = np.arctan2(DIRS[:, 1], DIRS[:, 0]) % (2 * np.pi)


def bucket_mid() -> NDArray[np.float64]:
    lo = THETA
    hi = np.roll(THETA, -1)
    hi = np.where(hi <= lo, hi + 2 * np.pi, hi)
    return (0.5 * (lo + hi)) % (2 * np.pi)


def bend_table(lam: float) -> NDArray[np.float64]:
    """bend[b, k] for k in 0..2 = moving to bucket b-1, b, b+1."""
    mid = bucket_mid()
    out = np.zeros((NB, 3))
    for b in range(NB):
        for k, b2 in enumerate(((b - 1) % NB, b, (b + 1) % NB)):
            d = abs(mid[b2] - mid[b])
            d = min(d, 2 * np.pi - d)
            out[b, k] = lam * d * d / L0_M
    return out


def density(inside: NDArray[np.bool_], clearance: NDArray[np.float64], res: float,
            r0: float) -> NDArray[np.float64]:
    cl = np.where(inside & np.isfinite(clearance), clearance, 0.0)
    return np.where(cl > 0, 1.0 + (r0 / np.maximum(cl, res / 2.0)) ** 2, BUILDING_COST)


@dataclass(frozen=True, eq=False)
class StateGraph:
    node: NDArray[np.int64]                   # (ny, nx) -> compact id or -1
    nbr: NDArray[np.int32]                     # (N, 16) neighbour id or -1
    cost: NDArray[np.float64]                  # (N, 16)
    rr: NDArray[np.int64]                      # (N,) row of each compact id
    cc: NDArray[np.int64]                      # (N,) column of each compact id


def build_graph(inside: NDArray[np.bool_], clearance: NDArray[np.float64], res: float,
                 r0: float) -> StateGraph:
    ny, nx = inside.shape
    node: NDArray[np.int64] = -np.ones((ny, nx), dtype=np.int64)
    node[inside] = np.arange(int(inside.sum()))
    c = density(inside, clearance, res, r0)
    rr, cc = np.nonzero(inside)
    N = len(rr)
    nbr: NDArray[np.int32] = -np.ones((N, NB), dtype=np.int32)
    cost: NDArray[np.float64] = np.zeros((N, NB))

    def at(r: NDArray[np.int64],
           col: NDArray[np.int64]) -> tuple[NDArray[np.bool_], NDArray[np.int64],
                                             NDArray[np.int64]]:
        ok = (r >= 0) & (r < ny) & (col >= 0) & (col < nx)
        rv = np.where(ok, r, 0)
        cv = np.where(ok, col, 0)
        return ok & inside[rv, cv], rv, cv

    for d, (dx, dy) in enumerate(DIRS):
        touched = [(dy, dx)]
        if abs(dx) == 2:
            touched += [(0, int(np.sign(dx))), (dy, int(np.sign(dx)))]
        elif abs(dy) == 2:
            touched += [(int(np.sign(dy)), 0), (int(np.sign(dy)), dx)]
        ok = np.ones(N, dtype=bool)
        csum = c[rr, cc].copy()
        for (ddr, ddc) in touched:
            okk, rv, cv = at(rr + ddr, cc + ddc)
            ok &= okk
            csum = csum + np.where(okk, c[rv, cv], 0.0)
        _, rv, cv = at(rr + dy, cc + dx)
        step = res * math.hypot(dx, dy)
        nbr[ok, d] = node[rv[ok], cv[ok]]
        cost[ok, d] = step * csum[ok] / (1 + len(touched))
    return StateGraph(node=node, nbr=nbr, cost=cost, rr=rr.astype(np.int64),
                       cc=cc.astype(np.int64))


# ------------------------------------------------------------------ Dijkstra over (cell, bucket)

@njit
def _push(heap: NDArray[np.int32], pos: NDArray[np.int32], dist: NDArray[np.float64], n: int,
          s: int) -> int:
    heap[n] = s
    pos[s] = n
    i = n
    while i > 0:
        p = (i - 1) >> 1
        if dist[heap[p]] <= dist[heap[i]]:
            break
        a, b = heap[p], heap[i]
        heap[p], heap[i] = b, a
        pos[b], pos[a] = p, i
        i = p
    return n + 1


@njit
def _sift_up(heap: NDArray[np.int32], pos: NDArray[np.int32], dist: NDArray[np.float64],
             i: int) -> None:
    while i > 0:
        p = (i - 1) >> 1
        if dist[heap[p]] <= dist[heap[i]]:
            break
        a, b = heap[p], heap[i]
        heap[p], heap[i] = b, a
        pos[b], pos[a] = p, i
        i = p


@njit
def _pop(heap: NDArray[np.int32], pos: NDArray[np.int32], dist: NDArray[np.float64],
         n: int) -> tuple[int, int]:
    top = heap[0]
    n -= 1
    pos[top] = -2
    if n > 0:
        last = heap[n]
        heap[0] = last
        pos[last] = 0
        i = 0
        while True:
            lc = 2 * i + 1
            if lc >= n:
                break
            r = lc + 1
            m = lc if (r >= n or dist[heap[lc]] <= dist[heap[r]]) else r
            if dist[heap[i]] <= dist[heap[m]]:
                break
            a, b = heap[i], heap[m]
            heap[i], heap[m] = b, a
            pos[b], pos[a] = i, m
            i = m
    return top, n


@njit
def _dijkstra(roots: NDArray[np.int64], nbr: NDArray[np.int32], cost: NDArray[np.float64],
              bend: NDArray[np.float64], is_target: NDArray[np.bool_], n_targets: int,
              dist: NDArray[np.float64], pred: NDArray[np.int32], heap: NDArray[np.int32],
              pos: NDArray[np.int32], first: NDArray[np.int64]) -> None:
    """Multi-root Dijkstra over states node*16+b. `first[node]` = the first settled state of each
    target node (its best bucket). Stops once every target node is settled."""
    NBk = 16
    dist[:] = np.inf
    pred[:] = -1
    pos[:] = -1
    first[:] = -1
    n = 0
    for i in range(roots.shape[0]):
        for b in range(NBk):
            s = roots[i] * NBk + b
            if dist[s] > 0.0:
                dist[s] = 0.0
                n = _push(heap, pos, dist, n, s)
    left = n_targets
    while n > 0 and left > 0:
        s, n = _pop(heap, pos, dist, n)
        u = s // NBk
        b = s - u * NBk
        if is_target[u] and first[u] < 0:
            first[u] = s
            left -= 1
        du = dist[s]
        for k in range(3):
            b2 = (b + k - 1) % NBk
            extra = bend[b, k]
            for j in range(2):
                d = (b2 + j) % NBk
                v = nbr[u, d]
                if v < 0:
                    continue
                t = v * NBk + b2
                if pos[t] == -2:
                    continue
                nd = du + cost[u, d] + extra
                if nd < dist[t]:
                    dist[t] = nd
                    pred[t] = s
                    if pos[t] == -1:
                        n = _push(heap, pos, dist, n, t)
                    else:
                        _sift_up(heap, pos, dist, pos[t])


@njit
def _credit(first: NDArray[np.int64], pred: NDArray[np.int32], targets: NDArray[np.int64],
            tw: NDArray[np.float64], src_node: int, scale: float, stamp: NDArray[np.int64],
            it0: int, out: NDArray[np.float64], rr: NDArray[np.int64], cc: NDArray[np.int64],
            grid: NDArray[np.int64]) -> int:
    """For each target node: walk its best route's pred chain and credit, once per route, every
    cell strictly between the root and the target -- the step cells AND the two cells a knight
    step crosses (a knight step would otherwise jump a cell, leave holes in the field and cross a
    one-cell wall uncredited). Returns the next stamp id."""
    NBk = 16
    it = it0
    for i in range(targets.shape[0]):
        t = targets[i]
        w = tw[i]
        if w == 0.0 or t == src_node or first[t] < 0:
            continue
        it += 1
        stamp[t] = it
        if src_node >= 0:
            stamp[src_node] = it
        s = first[t]
        while True:
            p = pred[s]
            if p < 0:
                break
            v = s // NBk
            u = p // NBk
            dr = rr[v] - rr[u]
            dc = cc[v] - cc[u]
            if abs(dc) == 2 or abs(dr) == 2:
                if abs(dc) == 2:
                    sc = 1 if dc > 0 else -1
                    m1 = grid[rr[u], cc[u] + sc]
                    m2 = grid[rr[u] + dr, cc[u] + sc]
                else:
                    sr = 1 if dr > 0 else -1
                    m1 = grid[rr[u] + sr, cc[u]]
                    m2 = grid[rr[u] + sr, cc[u] + dc]
                if m1 >= 0 and stamp[m1] != it:
                    stamp[m1] = it
                    out[m1] += scale * w
                if m2 >= 0 and stamp[m2] != it:
                    stamp[m2] = it
                    out[m2] += scale * w
            if pred[p] >= 0 and stamp[u] != it:     # u is not the root: credit it
                stamp[u] = it
                out[u] += scale * w
            s = p
    return it


# ------------------------------------------------------------------ variants

def _alloc(N: int) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32],
                             NDArray[np.int32], NDArray[np.int64], NDArray[np.int64]]:
    M = N * NB
    return (np.empty(M), np.empty(M, np.int32), np.empty(M, np.int32), np.empty(M, np.int32),
            np.empty(N, np.int64), np.zeros(N, np.int64))


@dataclass(frozen=True, eq=False)
class _PairWork:
    nbr: NDArray[np.int32]
    cost: NDArray[np.float64]
    bend: NDArray[np.float64]
    tnodes: NDArray[np.int64]
    tw: NDArray[np.float64]
    is_target: NDArray[np.bool_]
    rr: NDArray[np.int64]
    cc: NDArray[np.int64]
    grid: NDArray[np.int64]


_WORK: _PairWork | None = None


def _pairs_chunk(srcs: list[tuple[int, int]]) -> NDArray[np.float64]:
    w = _WORK
    assert w is not None, "_pairs_chunk called with no _WORK set"
    N = w.nbr.shape[0]
    dist, pred, heap, pos, first, stamp = _alloc(N)
    out = np.zeros(N)
    it = 0
    for s, mult in srcs:
        _dijkstra(np.array([s], np.int64), w.nbr, w.cost, w.bend, w.is_target,
                  int(w.is_target.sum()), dist, pred, heap, pos, first)
        it = _credit(first, pred, w.tnodes, w.tw, s, float(mult), stamp, it, out, w.rr, w.cc,
                     w.grid)
    return out


def pair_counts(g: StateGraph, bend: NDArray[np.float64], home_nodes: NDArray[np.int64],
                 source_homes: NDArray[np.int64], workers: int) -> NDArray[np.float64]:
    """All pairs: per cell, (source, target) pairs whose best routes pass through it.
    `source_homes` lists source home nodes (with repeats for homes sharing a cell)."""
    if workers > 1 and multiprocessing.current_process().daemon:
        raise RuntimeError(
            f"pair_counts(workers={workers}) cannot fork: this process is a daemonic pool worker, "
            f"and daemonic processes may not have children. Parallelize across blocks OR across "
            f"sources, not both -- set workers=1 on the betweenness desire source for runs that "
            f"already fork per block.")
    tnodes, tw = np.unique(home_nodes, return_counts=True)
    is_target = np.zeros(g.nbr.shape[0], dtype=np.bool_)
    is_target[tnodes] = True
    snodes, smult = np.unique(source_homes, return_counts=True)
    global _WORK
    _WORK = _PairWork(nbr=g.nbr, cost=g.cost, bend=bend, tnodes=tnodes.astype(np.int64),
                       tw=tw.astype(np.float64), is_target=is_target, rr=g.rr, cc=g.cc,
                       grid=g.node)
    jobs = list(zip(snodes.tolist(), smult.tolist(), strict=True))
    if workers <= 1 or len(jobs) < 16:
        return _pairs_chunk(jobs)
    parts = [jobs[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("fork")) as ex:
        return np.sum(list(ex.map(_pairs_chunk, parts)), axis=0)


def egress_counts(g: StateGraph, bend: NDArray[np.float64], home_nodes: NDArray[np.int64],
                   band_nodes: NDArray[np.int64]) -> NDArray[np.float64]:
    """Egress: every street-band cell is a root in every bucket; per cell, how many homes' best
    routes to the street pass through it."""
    tnodes, tw = np.unique(home_nodes, return_counts=True)
    is_target = np.zeros(g.nbr.shape[0], dtype=np.bool_)
    is_target[tnodes] = True
    N = g.nbr.shape[0]
    dist, pred, heap, pos, first, stamp = _alloc(N)
    out = np.zeros(N)
    _dijkstra(np.asarray(band_nodes, np.int64), g.nbr, g.cost, bend, is_target, len(tnodes),
              dist, pred, heap, pos, first)
    _credit(first, pred, tnodes.astype(np.int64), tw.astype(np.float64), -1, 1.0, stamp, 0, out,
            g.rr, g.cc, g.node)
    return out


def to_grid(values: NDArray[np.float64], g: StateGraph,
            inside: NDArray[np.bool_]) -> NDArray[np.float64]:
    f = np.full(g.node.shape, np.nan)
    f[inside] = values[g.node[inside]]
    return f


# ------------------------------------------------------------------ checks

def route_cost(inside: NDArray[np.bool_], clearance: NDArray[np.float64], res: float, r0: float,
                lam: float, a: tuple[int, int],
                b: tuple[int, int]) -> tuple[float, NDArray[np.int64]]:
    g = build_graph(inside, clearance, res, r0)
    node = g.node
    N = g.nbr.shape[0]
    dist, pred, heap, pos, first, _stamp = _alloc(N)
    is_t = np.zeros(N, np.bool_)
    is_t[node[b]] = True
    _dijkstra(np.array([node[a]], np.int64), g.nbr, g.cost, bend_table(lam), is_t, 1,
              dist, pred, heap, pos, first)
    cells = []
    s = first[node[b]]
    rc = np.argwhere(node >= 0)
    while s >= 0:
        cells.append(s // NB)
        s = pred[s]
    return float(dist[first[node[b]]]), rc[np.array(cells[::-1])]
