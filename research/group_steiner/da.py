"""Dual ascent (Wong 1984) for the directed-cut formulation in gst.formulation, plus reduced-cost
reductions. Scales where the cut LP does not.

Dual ascent: for each terminal not yet joined to the root by zero-reduced-cost arcs, grow W = the
nodes that reach it over zero arcs; while W holds no root-connected node, raise the dual of the
cut into W by the least reduced cost on it (every such cut is valid: root outside, terminal
inside), lower those arcs' reduced costs, and absorb the tails that hit zero. The sum of raises is
a valid lower bound at every moment, and reduced costs stay >= 0.

Reductions (Polzin & Daneshmand): any solution using arc (u, v) costs at least
LB + d(root, u) + cbar(u, v) + d(v, T) in reduced costs; if that exceeds a known upper bound the
arc is in no better solution and is dropped. Same for a node with LB + d(root, v) + d(v, T).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.sparse.csgraph import dijkstra

EPS = 1e-9


def _csr(n: int, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(key, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, key + 1, 1)
    return np.cumsum(ptr), order.astype(np.int64)


@njit(cache=True)
def _refresh_rootconn(root, head, cbar, out_ptr, out_arcs, rootconn, stack):
    """Forward zero-arc closure from the root."""
    rootconn[:] = False
    rootconn[root] = True
    top = 0
    stack[top] = root
    top += 1
    while top > 0:
        top -= 1
        u = stack[top]
        for k in range(out_ptr[u], out_ptr[u + 1]):
            a = out_arcs[k]
            v = head[a]
            if cbar[a] <= EPS and not rootconn[v]:
                rootconn[v] = True
                stack[top] = v
                top += 1


@njit(cache=True)
def _dual_ascent(n, root, tail, head, cbar, in_ptr, in_arcs, out_ptr, out_arcs, terminals):
    lb = 0.0
    inw = np.zeros(n, np.int64)                # stamp: node in W of the current terminal
    rootconn = np.zeros(n, np.bool_)
    stack = np.empty(n, np.int64)
    wlist = np.empty(n, np.int64)
    _refresh_rootconn(root, head, cbar, out_ptr, out_arcs, rootconn, stack)
    stamp = 0
    for t in terminals:
        if rootconn[t]:
            continue
        stamp += 1
        nw = 0
        inw[t] = stamp
        wlist[nw] = t
        nw += 1
        # reverse zero closure of {t}
        top = 0
        stack[top] = t
        top += 1
        joined = False
        while True:
            while top > 0:
                top -= 1
                w = stack[top]
                if rootconn[w]:
                    joined = True
                for k in range(in_ptr[w], in_ptr[w + 1]):
                    a = in_arcs[k]
                    u = tail[a]
                    if cbar[a] <= EPS and inw[u] != stamp:
                        inw[u] = stamp
                        wlist[nw] = u
                        nw += 1
                        stack[top] = u
                        top += 1
            if joined:
                break
            delta = np.inf
            for i in range(nw):
                w = wlist[i]
                for k in range(in_ptr[w], in_ptr[w + 1]):
                    a = in_arcs[k]
                    if inw[tail[a]] != stamp and cbar[a] < delta:
                        delta = cbar[a]
            if not np.isfinite(delta):
                return -1.0                      # infeasible: t unreachable from the root
            lb += delta
            for i in range(nw):
                w = wlist[i]
                for k in range(in_ptr[w], in_ptr[w + 1]):
                    a = in_arcs[k]
                    u = tail[a]
                    if inw[u] != stamp:
                        cbar[a] -= delta
                        if cbar[a] <= EPS:
                            cbar[a] = 0.0
                            inw[u] = stamp
                            wlist[nw] = u
                            nw += 1
                            stack[top] = u
                            top += 1
        _refresh_rootconn(root, head, cbar, out_ptr, out_arcs, rootconn, stack)
    return lb


def dual_ascent(f) -> tuple[float, np.ndarray]:
    """(lower bound, reduced costs per arc) for formulation `f` (gst.Formulation)."""
    n = f.n_nodes
    in_ptr, in_arcs = _csr(n, f.head)
    out_ptr, out_arcs = _csr(n, f.tail)
    cbar = f.cost.astype(np.float64).copy()
    order = np.asarray(f.terminals, dtype=np.int64)
    lb = _dual_ascent(n, 0, f.tail.astype(np.int64), f.head.astype(np.int64), cbar,
                      in_ptr, in_arcs, out_ptr, out_arcs, order)
    if lb < 0:
        raise RuntimeError("a group is unreachable from the street")
    return float(lb), cbar


def reduced_cost_keep(f, lb: float, cbar: np.ndarray, ub: float) -> np.ndarray:
    """Mask of arcs that can still appear in a solution cheaper than `ub`."""
    n = f.n_nodes
    g = sp.csr_matrix((np.maximum(cbar, 1e-12), (f.tail, f.head)), shape=(n, n))
    d_root = dijkstra(g, indices=0)
    d_term = dijkstra(g.T.tocsr(), indices=np.asarray(f.terminals), min_only=True)
    bound = lb + d_root[f.tail] + cbar + d_term[f.head]
    return bound <= ub + 1e-6 * max(ub, 1.0)


@njit(cache=True)
def _propagate(head, cbar, out_ptr, out_arcs, rootconn, stack, starts, nstarts):
    """Extend the root's forward zero closure from newly root-connected nodes."""
    top = 0
    for i in range(nstarts):
        stack[top] = starts[i]
        top += 1
    while top > 0:
        top -= 1
        u = stack[top]
        for k in range(out_ptr[u], out_ptr[u + 1]):
            a = out_arcs[k]
            v = head[a]
            if cbar[a] <= EPS and not rootconn[v]:
                rootconn[v] = True
                stack[top] = v
                top += 1


@njit(cache=True)
def _dual_ascent_rr(n, root, tail, head, cbar, in_ptr, in_arcs, out_ptr, out_arcs, terminals,
                    rec_nodes, rec_offs):
    """Round-robin: each pass gives every still-active terminal ONE raise of its current cut.
    Each raise's W is appended to rec_nodes (rec_offs[i]..rec_offs[i+1]) while capacity lasts;
    rec_offs[0] holds the number of recorded raises on return (offsets start at index 1)."""
    lb = 0.0
    nrec = 0
    pos = 0
    cap_nodes = len(rec_nodes)
    cap_recs = len(rec_offs) - 2
    rec_offs[1] = 0
    inw = np.zeros(n, np.int64)
    rootconn = np.zeros(n, np.bool_)
    stack = np.empty(n, np.int64)
    stack2 = np.empty(n, np.int64)
    wlist = np.empty(n, np.int64)
    starts = np.empty(n, np.int64)
    _refresh_rootconn(root, head, cbar, out_ptr, out_arcs, rootconn, stack)
    active = np.ones(len(terminals), np.bool_)
    stamp = 0
    n_active = len(terminals)
    while n_active > 0:
        for ti in range(len(terminals)):
            if not active[ti]:
                continue
            t = terminals[ti]
            stamp += 1
            nw = 0
            inw[t] = stamp
            wlist[nw] = t
            nw += 1
            top = 0
            stack[top] = t
            top += 1
            joined = False
            while top > 0:
                top -= 1
                w = stack[top]
                if rootconn[w]:
                    joined = True
                    break
                for k in range(in_ptr[w], in_ptr[w + 1]):
                    a = in_arcs[k]
                    u = tail[a]
                    if cbar[a] <= EPS and inw[u] != stamp:
                        inw[u] = stamp
                        wlist[nw] = u
                        nw += 1
                        stack[top] = u
                        top += 1
            if joined:
                active[ti] = False
                n_active -= 1
                continue
            delta = np.inf
            for i in range(nw):
                w = wlist[i]
                for k in range(in_ptr[w], in_ptr[w + 1]):
                    a = in_arcs[k]
                    if inw[tail[a]] != stamp and cbar[a] < delta:
                        delta = cbar[a]
            if not np.isfinite(delta):
                return -1.0
            lb += delta
            if nrec < cap_recs and pos + nw <= cap_nodes:
                for i in range(nw):
                    rec_nodes[pos + i] = wlist[i]
                pos += nw
                nrec += 1
                rec_offs[nrec + 1] = pos
            ns = 0
            for i in range(nw):
                w = wlist[i]
                for k in range(in_ptr[w], in_ptr[w + 1]):
                    a = in_arcs[k]
                    u = tail[a]
                    if inw[u] != stamp:
                        cbar[a] -= delta
                        if cbar[a] <= EPS:
                            cbar[a] = 0.0
                            if rootconn[u] and not rootconn[w]:
                                rootconn[w] = True
                                starts[ns] = w
                                ns += 1
            if ns > 0:
                _propagate(head, cbar, out_ptr, out_arcs, rootconn, stack2, starts, ns)
    rec_offs[0] = nrec
    return lb


def dual_ascent_rr(f, record: bool = False, cap_nodes: int = 30_000_000,
                   cap_recs: int = 2_000_000):
    """(lower bound, reduced costs) -- and, with `record`, the cut arc sets of every raise
    (deduplicated), a valid starting pool for the cut LP that already proves the DA bound."""
    n = f.n_nodes
    in_ptr, in_arcs = _csr(n, f.head)
    out_ptr, out_arcs = _csr(n, f.tail)
    cbar = f.cost.astype(np.float64).copy()
    rec_nodes = np.empty(cap_nodes if record else 1, np.int64)
    rec_offs = np.zeros((cap_recs if record else 0) + 2, np.int64)
    lb = _dual_ascent_rr(n, 0, f.tail.astype(np.int64), f.head.astype(np.int64), cbar,
                         in_ptr, in_arcs, out_ptr, out_arcs, np.asarray(f.terminals, np.int64),
                         rec_nodes, rec_offs)
    if lb < 0:
        raise RuntimeError("a group is unreachable from the street")
    if not record:
        return float(lb), cbar
    nrec = int(rec_offs[0])
    offs = rec_offs[1:nrec + 2]
    cuts, seen = [], set()
    inw = np.zeros(n, bool)
    for i in range(nrec):
        w = rec_nodes[offs[i]:offs[i + 1]]
        inw[w] = True
        arcs = np.flatnonzero(~inw[f.tail] & inw[f.head])
        inw[w] = False
        key = arcs.tobytes()
        if key not in seen:
            seen.add(key)
            cuts.append(arcs)
    return float(lb), cbar, cuts
