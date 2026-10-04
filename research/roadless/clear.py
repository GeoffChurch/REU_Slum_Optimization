"""Road-free clearing: remove whole buildings, greedily, by the roadless score.

Tension. Solve once with the remaining buildings as a poor conductor (open fraction eps
wherever a footprint is), so the potential is defined inside them. dP/dw_e = -(u_a - u_b)^2, so
a building's first-order gain from clearing is the sum over the edges it covers of
(Delta u)^2 x (the weight gained). That is the TENSION: how hard the escape flow presses on it.

Greedy. Each step: one tension solve (two for power != 1), then a Picker clears: Screened (exact
solves for the top M by tension, clear the best) or Batched (a spaced set of the top buildings up
to the next multiple of delta of population, one exact solve). `loo` checks the screen against
exact leave-one-out on one step.

    PYTHONPATH=. pixi run python research/roadless/clear.py loo <block idx> [h]
    PYTHONPATH=. pixi run python research/roadless/clear.py run <workers> <picker> <h> <d_max> <pop> <p> <along> <solver>
    PYTHONPATH=. pixi run python research/roadless/clear.py some <workers> <picker> <h> <d_max> <pop> <p> <along> <solver> <ids,>

picker: M4 | B0.01g3 | S0.01cat; along: uni | sl<beta> | ss<beta>k<kappa>[n<hill>r<r0>];
solver: cpu | gpu, the device of the solves, sightline scans and catchment sweep (gpu needs
CUDA_PATH, e.g. /usr)
"""
from __future__ import annotations

import dataclasses
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

EPS = 0.01
RTOL_TENSION = 1e-3     # a ranking: checked unchanged against 1e-9 (NOTES, "Speed")
RTOL_SCORE = 1e-5       # P is quadratic in u: this gives it to ~1e-10


def _cells_axes(sy: lifted.System, x, K: int):
    """Per-unknown device vector -> (free cells, K), zero on ground."""
    out = sy.xp.zeros((len(sy.unk_cell), K))
    out[sy.unk_cell >= 0] = x.reshape(-1, K)
    return out


class Tension:
    """One eps-material solve's worth of state, on the solver's device: the system, primal u and
    adjoint lam per unknown, the edges (ca, cb, ka, kb, dw, w) with their weight now and the
    weight clearing everything would add at the current along-factors, and the attribution
    att = bfree x diag(scale) (buildings x free cells; in the eps world the free cells are the
    inside cells), the share of each cell's gain a building owns. g (host) = the LOCAL
    first-order gain (a building's own edges, beta_e = (att[a] + att[b]) / 2 of each) + the
    NONLOCAL gain (the along-factors' response elsewhere, e.g. straight runs lengthening through
    the cleared cells)."""

    def __init__(self, system: lifted.System, u, lam, fam, bfree, scale, K: int, home_u,
                 removed, nonlocal_gain, solver: lifted.Solver):
        self.system, self.u, self.lam, self.fam, self.K = system, u, lam, fam, K
        self.bfree, self.scale, self.solver = bfree, scale, solver
        self.home_u = home_u                  # per building (host), from the eps solve
        self.g_local = self._gain()
        self.g = self.g_local + nonlocal_gain
        self.g[removed] = -np.inf

    def _gain(self) -> np.ndarray:
        xp = self.system.xp
        uk = _cells_axes(self.system, self.u, self.K)
        lk = _cells_axes(self.system, self.lam, self.K)
        nc = uk.shape[0]
        T = xp.zeros(nc)
        for ca, cb, ka, kb, dw, _w in self.fam:
            q = dw * (uk[ca, ka] - uk[cb, kb]) * (lk[ca, ka] - lk[cb, kb])
            T += 0.5 * (xp.bincount(ca, q, minlength=nc) + xp.bincount(cb, q, minlength=nc))
        return self.solver.to_host(self.bfree @ (self.scale * T))


class Clearing:
    def __init__(self, block, h: float, p: lifted.Params, population=None,
                 rtol: float = RTOL_TENSION, search: lifted.AlongConductance | None = None):
        """`search`: the along-conductance the TENSION ranks under (e.g. more translucent
        buildings, so counterfactual corridors show in the gradient); scoring stays `p`."""
        self.sc = common.Scorer(block, h, p, population=population)
        self.cost = self.sc.w / self.sc.w.sum()        # population share of each building
        self.p = p
        self.ps = p if search is None else dataclasses.replace(p, along=search)
        self.rtol = rtol                               # for the tension solves
        g = self.sc.grid
        xp = p.solver.xp
        self.full = g.isub.mean(axis=-1)               # everything open
        # sub-sample -> building index (-1 = none); overlaps go to the later footprint
        self.lab = lab = g.label_sub(self.sc.polys)
        self.n = len(self.sc.polys)
        # bfree (buildings x inside cells, row-major): the fraction of each cell a building covers
        S2 = lab.shape[-1]
        flat = lab.reshape(-1, S2)
        cells, subs = np.nonzero(flat >= 0)
        inside_id = -np.ones(g.inside.size, dtype=np.int64)
        inside_id[g.inside.ravel()] = np.arange(int(g.inside.sum()))
        bf = sp.coo_matrix((np.full(len(cells), 1.0 / S2), (flat[cells, subs], inside_id[cells])),
                           shape=(self.n, int(g.inside.sum()))).tocsr()
        self.bf = bf                                   # host copy
        self.bfree = p.solver.sparse.csr_matrix(bf)
        self.inside_flat = xp.asarray(np.flatnonzero(g.inside))
        self.removed = np.zeros(self.n, dtype=bool)

    def open_(self, removed: np.ndarray) -> np.ndarray:
        g = self.sc.grid
        cleared = (self.lab >= 0) & removed[np.maximum(self.lab, 0)]
        return (g.isub & (~g.bsub | cleared)).mean(axis=-1)

    def P(self, removed: np.ndarray) -> float:
        return self.sc.P_free(self.open_(removed))

    def tension(self, power: float = 1.0, restore: bool = False) -> Tension:
        """First-order gain in J_power of clearing each (remaining) building, from one
        eps-material solve (power 1: the potential is its own adjoint) or two (power != 1: the
        adjoint L lam = dJ/du = power u_i^(power-1) x injection). dJ/dw_e = -(du_e)(dlam_e).
        `restore`: instead, of putting back each cleared building (negative: J rises), by the
        same finite difference toward everything restored (a derivative at the 0/1 clearing
        splits each edge among the many cells tied at the min of fully open space and
        understates it). Everything but the per-building results stays on the device."""
        g = self.sc.grid
        xp, th = self.p.solver.xp, self.p.solver.to_host
        op = self.open_(self.removed)
        op_eps = op + EPS * (self.full - op)
        sy = lifted.System(g, op_eps, self.ps)
        sol = lifted.solve(g, op_eps, self.sc.f, self.ps, system=sy, rtol=self.rtol, host=False)
        hsol = dataclasses.replace(sol, u=th(sol.u), cell=th(sol.cell),
                                   unk_cell=th(sol.unk_cell))
        home_u = self.sc.home_u_of(hsol, op_eps)
        if power == 1.0:
            lam = sol.u
        else:
            mult = np.zeros_like(self.sc.f)
            on = self.sc.owner >= 0
            mult[on] = power * np.nan_to_num(home_u[self.sc.owner[on]]) ** (power - 1.0)
            lam = lifted.solve(g, op_eps, self.sc.f * mult, self.ps, system=sy, rtol=self.rtol,
                               host=False).u
        # Each edge's weight gained by clearing everything, at the CURRENT along-factors (the
        # local term): w = w_uniform x min(factor at the two ends). The factors' own response to
        # clearing (straight runs lengthening through the cleared cells, boosting edges
        # elsewhere) is the nonlocal term, from the along-conductance's vjp below.
        K = self.p.K
        pu = dataclasses.replace(self.ps, along=lifted.Uniform())
        Fl = self.ps.along.layers(op_eps, g.h, K)
        uk = _cells_axes(sy, sol.u, K)
        lk = _cells_axes(sy, lam, K)
        ff = self.inside_flat                            # free cell -> flat (eps: all inside)
        Agrad = xp.zeros((K, g.inside.size))
        fam = []
        if restore:                                     # everything put back (eps world)
            op0 = self.open_(np.zeros(self.n, dtype=bool))
            ref = op0 + EPS * (self.full - op0)
        else:
            ref = self.full
        for i, ((ca, cb, ka, kb, wf), (ca2, cb2, _, _, wc)) in enumerate(zip(
                lifted.edges(g, ref, pu), lifted.edges(g, op_eps, pu), strict=True)):
            assert bool((ca == ca2).all()) and bool((cb == cb2).all())
            if i < K and not np.isscalar(Fl[i]):        # along family, layer i
                Fi = xp.asarray(Fl[i]).ravel()
                fa, fb = Fi[ff[ca]], Fi[ff[cb]]
                boost = xp.minimum(fa, fb)
                q = (uk[ca, ka] - uk[cb, kb]) * (lk[ca, ka] - lk[cb, kb])    # -dJ/dw_e
                Agrad[i] += xp.bincount(xp.where(fa <= fb, ff[ca], ff[cb]), q * wc,
                                        minlength=g.inside.size)
                fam.append((ca, cb, ka, kb, (wf - wc) * boost, wc * boost))
            elif i < K:
                fam.append((ca, cb, ka, kb, (wf - wc) * Fl[i], wc * Fl[i]))
            else:
                fam.append((ca, cb, ka, kb, wf - wc, wc))
        # nonlocal gain: d(gain)/d(open) at every cell, x the open each building would add
        dgain = xp.asarray(self.ps.along.vjp(op_eps, g.h, K, Agrad.reshape(K, *op_eps.shape)))
        nonlocal_gain = th(self.bfree @ dgain.ravel()[ff]) * (1.0 - EPS)
        # attribution: building j owns frac/(1 - op) of each cell it covers (restore: frac /
        # (op - op0), its share of the cell's cleared part)
        if restore:
            nonlocal_gain = -nonlocal_gain
            scale = 1.0 / xp.maximum(xp.asarray(op - op0).ravel()[ff], 1e-9)
        else:
            scale = 1.0 / xp.maximum(1.0 - xp.asarray(op).ravel()[ff], 1e-9)
        return Tension(system=sy, u=sol.u, lam=lam, fam=fam, bfree=self.bfree, scale=scale,
                       K=K, home_u=home_u, removed=~self.removed if restore else self.removed,
                       nonlocal_gain=nonlocal_gain, solver=self.p.solver)

def loo(i: int, h: float, pop: str = "count", power: float = 1.0) -> None:
    from scipy.stats import spearmanr
    blocks = common.build_blocks(common.recipients())
    b = blocks[i]
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8), population=common.POPULATIONS[pop])
    sc = c.sc
    P0 = sc.J(sc.u0, power)
    t = time.time()
    T = c.tension(power).g
    tt = time.time() - t
    t = time.time()
    exact = np.zeros(c.n)
    for j in range(c.n):
        r = c.removed.copy()
        r[j] = True
        op = c.open_(r)
        exact[j] = P0 - sc.J(sc.home_u_of(lifted.solve(sc.grid, op, sc.f, c.p), op), power)
    te = time.time() - t
    rho = spearmanr(T, exact).statistic
    top = np.argsort(-exact)
    rankT = np.argsort(np.argsort(-T))
    print(f"{b.block_id} n={c.n} h={h} pop={pop} p={power:g}: tension {tt:.0f}s, exact LOO {te:.0f}s "
          f"({te / c.n:.1f}s each)", flush=True)
    print(f"  Spearman(tension, exact gain) {rho:+.3f}; tension rank of the exact top 5: "
          f"{rankT[top[:5]].tolist()}; exact gain top 5 (share of P0): "
          f"{np.round(exact[top[:5]] / P0, 4).tolist()}", flush=True)
    for M in (1, 4, 8, 16):
        pick = np.argsort(-T)[:M]
        print(f"  M={M:2d}: best exact gain among tension top-M = "
              f"{exact[pick].max() / exact.max():.3f} of the true best", flush=True)


class Picked(NamedTuple):
    cleared: list[int]
    J: float          # J_power after clearing them
    P1: float         # P (= J_1) after clearing them


class Picker(Protocol):
    """What one greedy step clears, given the round's tension and J_power now."""
    name: str

    def pick(self, c: Clearing, t: Tension, J: float, power: float) -> Picked: ...


def _exact(c: Clearing, cleared: list[int], power: float) -> tuple[float, float]:
    r = c.removed.copy()
    r[cleared] = True
    op = c.open_(r)
    sol = lifted.solve(c.sc.grid, op, c.sc.f, c.p, rtol=RTOL_SCORE)
    return c.sc.J(c.sc.home_u_of(sol, op), power), sol.P


def _next_target(c: Clearing, delta: float) -> tuple[float, float]:
    """(D now, the next multiple of delta above it)."""
    D = float(c.cost[c.removed].sum())
    return D, (np.floor(D / delta + 1e-9) + 1) * delta


@dataclass(frozen=True)
class Screened:
    """Exact-solve the top M by tension, clear the one with the best exact gain per unit
    population. M + 1 (or 2) solves per building."""
    M: int

    @property
    def name(self) -> str:
        return f"M{self.M}"

    def pick(self, c: Clearing, t: Tension, J: float, power: float) -> Picked:
        T = t.g / c.cost
        best = Picked([], np.inf, np.inf)
        bestv = -np.inf
        for j in (int(j) for j in np.argsort(-T)[:self.M] if not c.removed[j]):
            Jj, P1 = _exact(c, [j], power)
            v = (J - Jj) / c.cost[j]
            if v > bestv:
                best, bestv = Picked([j], Jj, P1), v
        return best


@dataclass(frozen=True)
class Batched:
    """Clear, by tension per unit population, every building not within `gap_m` of one already
    taken this round, until D reaches the next multiple of `delta` (so a lens threshold on that
    lattice is overshot by at most one building, as in the one-at-a-time greedy). One exact
    solve per round, for the record; no screen."""
    delta: float
    gap_m: float

    @property
    def name(self) -> str:
        return f"B{self.delta:g}g{self.gap_m:g}"

    def pick(self, c: Clearing, t: Tension, J: float, power: float) -> Picked:
        T = t.g / c.cost
        D, target = _next_target(c, self.delta)
        taken: list[int] = []
        near: set[int] = set()
        for j in (int(j) for j in np.argsort(-T) if not c.removed[j]):
            if D >= target - 1e-12:
                break
            if j in near:
                continue
            taken.append(j)
            D += float(c.cost[j])
            near.update(int(k) for k in c.sc.tree.query(c.sc.polys[j], predicate="dwithin",
                                                         distance=self.gap_m))
        return Picked(taken, *_exact(c, taken, power))


class GramSource(Protocol):
    """How alike the candidates' effects are: a PSD matrix over `cand` whose normalised entries
    are the correlations rho_ij that `Spread` builds its batch from."""
    name: str

    def gram(self, c: Clearing, t: Tension, cand: list[int], power: float) -> np.ndarray: ...


class Sweep(Protocol):
    """Where Catchment's downhill sweep runs. prepare(A, u, rowsum) -> run(rows, cols, C): the
    (n x C) device array out[x, c] = share of node x's outflow that passes through column c's
    member nodes (the (rows, cols) pairs) on the way to ground. Flow only runs downhill, so out
    at x needs out at x's lower neighbours first. columns(n): columns per run on n nodes, so
    that the (n x C) result fits in memory."""

    def columns(self, n: int) -> int: ...

    def prepare(self, A, u, rowsum) -> Callable: ...


@njit(cache=True)
def _catch(indptr, indices, data, u, rowsum, order, inA, out):
    """out[x, c] = share of node x's outflow that passes through candidate c's nodes (inA) on the
    way to ground, sweeping nodes by increasing potential (flow only runs downhill)."""
    C = inA.shape[1]
    acc = np.zeros(C)
    for x in order:
        acc[:] = 0.0
        Q = rowsum[x] * u[x]                                  # straight to ground
        for p in range(indptr[x], indptr[x + 1]):
            y = indices[p]
            if y != x and u[y] < u[x]:
                q = -data[p] * (u[x] - u[y])
                Q += q
                for c in range(C):
                    acc[c] += q * out[y, c]
        for c in range(C):
            if inA[x, c]:
                out[x, c] = 1.0
            elif Q > 0:
                out[x, c] = acc[c] / Q


@dataclass(frozen=True)
class CpuSweep:
    """numba, one core: every node in order of increasing potential."""
    chunk: int = 16

    def columns(self, n: int) -> int:
        return self.chunk

    def prepare(self, A, u, rowsum):
        order = np.argsort(u, kind="stable")

        def run(rows, cols, C):
            inA = np.zeros((len(u), C), dtype=np.bool_)
            inA[rows, cols] = True
            out = np.zeros((len(u), C))
            _catch(A.indptr, A.indices, A.data, u, rowsum, order, inA, out)
            return out
        return run


_SWEEP_CU = r"""
extern "C" __global__ void indeg(const int* indptr, const int* indices, const double* u, int n,
                                 int* deg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    int d = 0;
    for (int p = indptr[x]; p < indptr[x + 1]; p++) {
        int y = indices[p];
        if (y != x && u[y] < u[x]) d++;
    }
    deg[x] = d;
}
// Kahn's algorithm, one wave per launch: settle order[start:end), release their upper neighbours
extern "C" __global__ void advance(const int* indptr, const int* indices, const double* u,
                                   int* order, int start, int end, int* deg, int* tail) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= end - start) return;
    int x = order[start + t];
    for (int p = indptr[x]; p < indptr[x + 1]; p++) {
        int y = indices[p];
        if (y != x && u[y] > u[x] && atomicSub(&deg[y], 1) == 1) order[atomicAdd(tail, 1)] = y;
    }
}
// one wave of the sweep: thread (i, c) for node order[start + i], column c; out is (n x C)
// column-major, the layout cuSPARSE's csr @ dense takes without a copy
extern "C" __global__ void catch_wave(const int* indptr, const int* indices, const double* data,
        const double* u, const double* rowsum, const int* order, int start, int cnt,
        const unsigned int* mask, int C, int n, double* out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)cnt * C) return;
    int i = (int)(t % cnt), c = (int)(t / cnt);
    int x = order[start + i];
    long long col = (long long)c * n, o = col + x;
    if ((mask[x] >> c) & 1u) { out[o] = 1.0; return; }
    double ux = u[x], Q = rowsum[x] * ux, acc = 0.0;
    for (int p = indptr[x]; p < indptr[x + 1]; p++) {
        int y = indices[p];
        if (y != x && u[y] < ux) {
            double q = -data[p] * (ux - u[y]);
            Q += q;
            acc += q * out[col + y];
        }
    }
    out[o] = Q > 0 ? acc / Q : 0.0;
}
"""


@dataclass(frozen=True)
class GpuSweep:
    """cupy: the downhill graph's topological waves (Kahn's algorithm, one launch per wave: a
    node is ready once all its lower neighbours are), then one launch per wave for the sweep,
    a thread per (node, column). Same arithmetic per node as the CPU sweep. `chunk` <= 32 (the
    member mask is one uint32 per node)."""
    chunk: int = 32
    _memo: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    def _kernels(self):
        import cupy as cp
        if "k" not in self._memo:
            mod = cp.RawModule(code=_SWEEP_CU)
            self._memo["k"] = tuple(mod.get_function(f) for f in ("indeg", "advance",
                                                                   "catch_wave"))
        return self._memo["k"]

    def columns(self, n: int) -> int:
        """Up to `chunk`, as many as a quarter of the free device memory holds: a block of
        several km^2 has ~50M unknowns, 32 columns of which are 11 GB. The pool's cached blocks
        are released first and not counted (they can be fragments of larger allocations: counted,
        22422 and 20543 under the sightline conductance asked for 8 GB that was not there)."""
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        free = cp.cuda.Device().mem_info[0]
        return max(1, min(self.chunk, int(free / 4 / (8 * n))))

    def prepare(self, A, u, rowsum):
        import cupy as cp
        k_deg, k_adv, k_wave = self._kernels()
        n = len(u)
        ip, ix = A.indptr.astype(cp.int32), A.indices.astype(cp.int32)
        deg = cp.empty(n, dtype=cp.int32)
        k_deg(((n + 255) // 256,), (256,), (ip, ix, u, np.int32(n), deg))
        first = cp.flatnonzero(deg == 0).astype(cp.int32)
        order = cp.empty(n, dtype=cp.int32)
        order[:len(first)] = first
        tail = cp.array([len(first)], dtype=cp.int32)
        waves = [0, len(first)]
        while waves[-1] < n:
            s, e = waves[-2], waves[-1]
            k_adv(((e - s + 255) // 256,), (256,), (ip, ix, u, order, np.int32(s), np.int32(e),
                                                    deg, tail))
            nxt = int(tail.get()[0])
            if nxt == e:
                raise RuntimeError("downhill graph has a cycle")
            waves.append(nxt)

        def run(rows, cols, C):
            assert C <= 32
            mask = cp.zeros(n, dtype=cp.uint32)
            for m in range(C):
                r = rows[cols == m]
                mask[r] |= np.uint32(1 << m)
            out = cp.zeros((n, C), order="F")
            for s, e in zip(waves[:-1], waves[1:], strict=True):
                tot = (e - s) * C
                k_wave(((tot + 255) // 256,), (256,),
                       (ip, ix, A.data, u, rowsum, order, np.int32(s), np.int32(e - s), mask,
                        np.int32(C), np.int32(n), out))
            return out
        return run


@dataclass(frozen=True)
class Catchment:
    """No solves: trace the round's flow downhill and give each candidate the share of each
    home's flow that passes through its cells. H_ij = sum_h c_h v_i(h) v_j(h), with c_h the
    home's weight in J_power (w_h u_h^power). The sweep runs where `sweep` says."""
    sweep: Sweep

    @property
    def name(self) -> str:
        return "cat"

    def gram(self, c: Clearing, t: Tension, cand: list[int], power: float) -> np.ndarray:
        sy, K, xp = t.system, t.K, t.system.xp
        spm = t.solver.sparse
        A = sy.A
        keep = xp.flatnonzero(sy.keep)
        u = t.u[keep]
        rowsum = A @ xp.ones(A.shape[0])
        # per kept unknown: its home (the owner of its cell) and its injection
        owner_free = xp.asarray(c.sc.owner).ravel()[sy.cell.ravel() >= 0]
        live = sy.unk_cell >= 0
        owner_unk = xp.repeat(owner_free[live], K)[keep]
        b = sy.load(c.sc.f, c.p)[keep]
        home = xp.flatnonzero(owner_unk >= 0)
        inj = xp.bincount(owner_unk[home], b[home], minlength=c.n)
        Hown = spm.coo_matrix((b[home], (owner_unk[home], home)),
                              shape=(c.n, len(keep))).tocsr()
        # members: kept positions of each candidate's unknowns (eps world: free cell = inside)
        assert len(sy.unk_cell) == c.bfree.shape[1]
        pos = -xp.ones(sy.n, dtype=xp.int64)
        pos[keep] = xp.arange(len(keep))
        sub = c.bfree[xp.asarray(cand)].tocoo()
        un = sy.unk_cell[sub.col]
        ok = un >= 0
        idx = pos[(un[ok][:, None] * K + xp.arange(K)[None, :]).ravel()]
        col = xp.repeat(sub.row[ok], K)
        rows, cols = idx[idx >= 0], col[idx >= 0]
        run = self.sweep.prepare(A, u, rowsum)
        V = xp.zeros((c.n, len(cand)))
        step = self.sweep.columns(len(keep))
        for s0 in range(0, len(cand), step):
            C = min(step, len(cand) - s0)
            sel = (cols >= s0) & (cols < s0 + C)
            V[:, s0:s0 + C] = Hown @ run(rows[sel], cols[sel] - s0, C)
        V = xp.where(inj[:, None] > 0, V / xp.where(inj > 0, inj, 1.0)[:, None], 0.0)
        wt = xp.asarray(c.sc.w * np.nan_to_num(t.home_u) ** power)
        return t.solver.to_host((V * wt[:, None]).T @ V)


def _spread(gain: np.ndarray, cost: np.ndarray, H: np.ndarray, need: float) -> list[int]:
    """Greedy batch under a normalised quadratic model: with rho = H's correlations, candidate i
    is worth g_i - sum_{j in batch} rho_ij sqrt(g_i g_j) once the batch is chosen (a duplicate of
    a chosen one is worth 0, an independent one keeps its g_i, a complementary one, rho < 0,
    gains). Take the best worth per unit cost until the batch's cost reaches `need`."""
    d = np.sqrt(np.clip(np.diag(H), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(np.outer(d, d) > 0, H / np.outer(d, d), 0.0)
    sg = np.sqrt(np.clip(gain, 0.0, None))
    worth = gain.astype(float).copy()
    avail = np.ones(len(gain), dtype=bool)
    chosen: list[int] = []
    taken = 0.0
    while taken < need - 1e-12 and avail.any():
        i = int(np.argmax(np.where(avail, worth / cost, -np.inf)))
        chosen.append(i)
        avail[i] = False
        taken += float(cost[i])
        worth -= rho[:, i] * sg * sg[i]
    return chosen


@dataclass(frozen=True)
class Spread:
    """Shortlist the top candidates by tension per unit population up to `reach` times the
    round's population step, measure how alike their effects are with `source`, and take the
    batch `_spread` builds from their correlations."""
    delta: float
    source: GramSource
    reach: float = 3.0

    @property
    def name(self) -> str:
        return f"S{self.delta:g}{self.source.name}"

    def pick(self, c: Clearing, t: Tension, J: float, power: float) -> Picked:
        T = t.g / c.cost
        D, target = _next_target(c, self.delta)
        order = [int(j) for j in np.argsort(-T) if not c.removed[j]]
        cum = np.cumsum(c.cost[order])
        cand = order[:int(np.searchsorted(cum, self.reach * (target - D))) + 1]
        H = self.source.gram(c, t, cand, power)
        chosen = _spread(t.g[cand], c.cost[cand], H, target - D)
        taken = [cand[i] for i in chosen]
        del t, H                    # the round's system and hierarchy, before the scoring solve
        return Picked(taken, *_exact(c, taken, power))


def grow(c: Clearing, picker: Picker, power: float, d_max: float, J: float):
    """The greedy from c's current clearing (J_power J): each step ranks the remaining buildings
    by tension per unit of population and lets `picker` clear some, until d_max. Yields
    (Picked, D after) per step; c.removed is updated before each yield."""
    while c.cost[c.removed].sum() < d_max - 1e-12 and not c.removed.all():
        pk = picker.pick(c, c.tension(power), J, power)
        c.removed[pk.cleared] = True
        J = pk.J
        yield pk, float(c.cost[c.removed].sum())


def greedy_block(b, picker: Picker, h: float, d_max: float, out: Path, population,
                 power: float, along: lifted.AlongConductance, solver: lifted.Solver,
                 search: lifted.AlongConductance | None = None) -> None:
    """The greedy to d_max. Records perm (in J_power) and perm1 (the p = 1 score) at every
    step; `cleared` lists the step's buildings in pick order."""
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8, along=along, solver=solver),
                 population=population, search=search)
    sc = c.sc
    J0 = sc.J(sc.u0, power)
    P0 = sc.P0
    rows = [dict(block=b.block_id, n=c.n, step=0, D=0.0, perm=0.0, perm1=0.0, cleared=[],
                 P0=P0, t=0.0)]
    t0 = time.time()
    for step, (pk, D) in enumerate(grow(c, picker, power, d_max, J0), start=1):
        rows.append(dict(block=b.block_id, n=c.n, step=step, D=D,
                         perm=1 - (pk.J / J0) ** (1 / power), perm1=1 - pk.P1 / P0,
                         cleared=pk.cleared, P0=P0, t=time.time() - t0))
        if c.n > 1000:
            print(f"  {b.block_id} step {step} D {D:.3f} perm' {rows[-1]['perm']:.3f}"
                  f" {rows[-1]['t']:.0f}s", flush=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, out)
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={c.n} {len(rows) - 1} steps, perm' at "
          f"end {rows[-1]['perm']:.3f}  {time.time() - t0:.0f}s", flush=True)


_BLOCKS: list = []
_CFG: dict = {}


def _one(i: int) -> str | None:
    """The block's id if it failed (reported, and the run exits non-zero at the end)."""
    b = _BLOCKS[i]
    out = _CFG["dir"] / f"{b.block_id}.parquet"
    if out.exists():
        return None
    try:
        greedy_block(b, _CFG["picker"], _CFG["h"], _CFG["d_max"], out,
                     common.POPULATIONS[_CFG["pop"]], _CFG["power"], _CFG["along"], _CFG["solver"],
                     _CFG["search"])
    except Exception as e:
        ours = [f for f in traceback.extract_tb(e.__traceback__) if "roadless" in f.filename]
        where = " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(ours[-5:]))
        print(f"{b.block_id} FAILED {type(e).__name__}: {str(e)[:200]} at {where}", flush=True)
        return b.block_id
    return None


def cleared_through(g: pd.DataFrame, step: int) -> np.ndarray:
    """Buildings cleared in steps 1..step of one block's greedy rows, in clearing order."""
    g = g[(g.step >= 1) & (g.step <= step)].sort_values("step")
    return np.array([j for js in g.cleared for j in js], dtype=np.int64)


def sweep_of(spec: str) -> Sweep:
    """`cpu` -> CpuSweep(); `gpu` -> GpuSweep() (chosen with the solver)."""
    if spec == "cpu":
        return CpuSweep()
    if spec == "gpu":
        return GpuSweep()
    raise ValueError(f"unknown sweep {spec!r}")


def picker_of(spec: str, sweep: Sweep) -> Picker:
    """`M4` -> Screened(4); `B0.01g3` -> Batched(delta 0.01, gap 3 m); `S0.01cat` ->
    Spread(delta 0.01) with Catchment (its sweep on `sweep`'s device). (Impact, Sketch and the
    no-spacing null were measured and dominated: NOTES.md, "Batching"; Impact again on the GPU,
    "GPU-resident rounds".)"""
    if spec.startswith("M"):
        return Screened(int(spec[1:]))
    if spec.startswith("B"):
        delta, gap = spec[1:].split("g")
        return Batched(float(delta), float(gap))
    m = re.fullmatch(r"S([0-9.]+)cat", spec)
    if m:
        return Spread(float(m.group(1)), Catchment(sweep))
    raise ValueError(f"unknown picker {spec!r}")


def rows_dir(picker: str, h: float, pop: str, power: float = 1.0, along: str = "uni") -> Path:
    return HERE / (f"clear_rows_{picker}_h{h:g}" + ("" if pop == "count" else f"_{pop}")
                   + ("" if power == 1.0 else f"_p{power:g}")
                   + ("" if along == "uni" else f"_{along}"))


def run(workers: int, picker: Picker, h: float, d_max: float, pop: str, power: float,
        along: lifted.AlongConductance, solver: lifted.Solver,
        search: lifted.AlongConductance | None, ids: list[str] | None = None) -> None:
    """All 220 study blocks, largest first, or just `ids`. `search`: the tension's
    along-conductance when it differs from the scoring one (rows dir `<along>@<search>`)."""
    import multiprocessing
    global _BLOCKS, _CFG
    d = rows_dir(picker.name, h, pop, power,
                 along.name + ("" if search is None else f"@{search.name}"))
    d.mkdir(exist_ok=True)
    _CFG = dict(picker=picker, h=h, d_max=d_max, dir=d, pop=pop, power=power, along=along,
                solver=solver, search=search)
    _BLOCKS = common.build_blocks(common.recipients() if ids is None else ids)
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        failed = [f for f in pool.imap_unordered(_one, list(range(len(_BLOCKS)))[::-1]) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


if __name__ == "__main__":
    if sys.argv[1] == "loo":
        loo(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
            sys.argv[4] if len(sys.argv) > 4 else "count",
            float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
    elif sys.argv[1] in ("run", "some"):
        # run <workers> <picker> <h> <d_max> <pop> <p> <along>[@<search along>] <solver> [ids,]
        scans = lifted.scans_of(sys.argv[9])
        specs = sys.argv[8].split("@")
        run(int(sys.argv[2]), picker_of(sys.argv[3], sweep_of(sys.argv[9])), float(sys.argv[4]),
            float(sys.argv[5]),
            sys.argv[6], float(sys.argv[7]), lifted.along_of(specs[0], scans),
            lifted.solver_of(sys.argv[9]),
            lifted.along_of(specs[1], scans) if len(specs) == 2 else None,
            sys.argv[10].split(",") if sys.argv[1] == "some" else None)
