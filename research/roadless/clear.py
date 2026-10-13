"""Road-free clearing: remove whole buildings, greedily, by the roadless score.

Tension. Solve once with the remaining buildings as a poor conductor (open fraction eps
wherever a footprint is), so the potential is defined inside them. dP/dw_e = -(u_a - u_b)^2, so
a building's first-order gain from clearing is the sum over the edges it covers of
(Delta u)^2 x (the weight gained). That is the TENSION: how hard the escape flow presses on it.

Greedy. Each step: one tension solve (two for power != 1), then a search.greedy_round clears
(M: exact solves for the top M by tension, clear the best; B: a spaced set of the top buildings up
to the next multiple of delta of population; S: a catchment-diverse such set). `loo` checks the
screen against exact leave-one-out on one step.

Free first (B's first stage, `free_first`). Under B a clearing is scored first by the population
it leaves enclosed, so the optimizers (relax.py, polish_greedy.py) first clear crossings that
free the homes enclosed at baseline, cheapest per unit of population, found on a graph of the
open components and the buildings' pieces (lifted.regions) and checked by the metric's own test,
then hold them cleared through the search with the rest of the budget.

    PYTHONPATH=. uv run python research/roadless/clear.py loo <block idx> [h]
    PYTHONPATH=. uv run python research/roadless/clear.py run <workers> <picker> <h> <d_max> <pop> <p> <cond> <solver>
    PYTHONPATH=. uv run python research/roadless/clear.py some <workers> <picker> <h> <d_max> <pop> <p> <cond> <solver> <ids,>

picker: M4 | B0.01g3 | S0.01cat; cond: the conductance (lifted.conductance_of: uni+apwfs,
ss<beta>k<kappa>[n<hill>r<r0>]+apwfs, ...)[@<the tension's along-conductance>];
solver: cpu | gpu, the device of the solves, sightline scans and catchment sweep (gpu needs
CUDA_PATH, e.g. /usr)
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple, Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

if TYPE_CHECKING:
    import search as engine  # search imports clear at run time

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
    def __init__(self, block, mesh: lifted.MeshSpec, p: lifted.Params, population=None,
                 rtol: float = RTOL_TENSION, search: lifted.AlongConductance | None = None, *,
                 demand: common.Demand):
        """`search`: the along-conductance the TENSION ranks under (e.g. more translucent
        buildings, so counterfactual corridors show in the gradient); scoring stays `p`.
        `demand`: what each home injects (common.Demand); the cost stays the population's."""
        self._on(common.Scorer(block, mesh, p, population=population, demand=demand), p, rtol,
                 search)

    @classmethod
    def on_scorer(cls, sc: common.Scorer, *, rtol: float,
                  search: lifted.AlongConductance | None) -> Clearing:
        """A clearing over a scorer already built (its grid, physics `sc.p`, population and
        demand): no second baseline solve."""
        c = cls.__new__(cls)
        c._on(sc, sc.p, rtol, search)
        return c

    def _on(self, sc: common.Scorer, p: lifted.Params, rtol: float,
            search: lifted.AlongConductance | None) -> None:
        self.sc = sc
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
        if not self.ps.cells.pointwise:
            raise NotImplementedError(
                f"{self.ps.cells.name}: a cell factor that looks past its cell moves the edges "
                "around a building too, which the secant tension (each edge's gain attributed to "
                "the buildings covering its cells) does not see")
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
            lam = lifted.solve(g, op_eps, self.sc.adjoint_load(home_u, power), self.ps,
                               system=sy, rtol=self.rtol, host=False).u
        # Each edge's weight gained by clearing everything, at the CURRENT along-factors (the
        # local term): w = w_uniform x min(factor at the two ends). The factors' own response to
        # clearing (straight runs lengthening through the cleared cells, boosting edges
        # elsewhere) is the nonlocal term, from the along-conductance's vjp below.
        K = self.p.K
        pu = dataclasses.replace(self.ps, along=lifted.Uniform())
        Fl = self.ps.along.layers(g, op_eps, K)
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
        dgain = xp.asarray(self.ps.along.vjp(g, op_eps, K, Agrad.reshape(K, *op_eps.shape)))
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
    mesh = lifted.UniformMesh(h, offset=lifted.OFFSET)
    c = Clearing(b, mesh, lifted.Params(ell_m=3.0, K=8, cells=lifted.OpenFraction()),
                 population=common.POPULATIONS[pop], demand=common.PopulationDemand())
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


def exact(c: Clearing, removed: np.ndarray, power: float) -> tuple[float, float]:
    """(J_power, P) of the clearing `removed`: real geometry, the metric conductance,
    RTOL_SCORE. The one exact entry point (memprobe_greedy instruments it)."""
    op = c.open_(removed)
    sol = lifted.solve(c.sc.grid, op, c.sc.f, c.p, rtol=RTOL_SCORE)
    return c.sc.J(c.sc.home_u_of(sol, op), power), sol.P


def _next_target(c: Clearing, delta: float) -> tuple[float, float]:
    """(D now, the next multiple of delta above it)."""
    D = float(c.cost[c.removed].sum())
    return D, (np.floor(D / delta + 1e-9) + 1) * delta


FREE_STEP = 1e-12   # free_first: every step through the region graph costs this more, so no
                    # edge weighs 0 and ties go to the crossing through fewer regions


@dataclass(frozen=True, eq=False)
class RegionGraph:
    """free_first's graph at baseline (lifted.Regions, built by lifted.regions from the
    operator's own steps): regions 0 .. ncomp - 1 the components of open space, then the regions
    a clearing enters, each with its `owner` building (`owner`: region - ncomp -> building),
    adjacent (a -> b, both ways) where a clearing joins them; the street steps into `entry`."""
    ncomp: int
    owner: np.ndarray
    a: np.ndarray
    b: np.ndarray
    entry: np.ndarray
    R: sp.csr_matrix            # (grid cells flat, regions): the regions a home is freed
                                # through at each cell (lifted.Regions)

    @classmethod
    def of(cls, c: Clearing) -> RegionGraph:
        g = c.sc.grid
        open0 = (c.open_(np.zeros(c.n, dtype=bool)) > 0).ravel()
        rg = lifted.regions(g, open0, c.bf, c.lab.reshape(g.inside.size, -1), c.p)
        return cls(ncomp=rg.ncomp, owner=rg.owner, a=rg.a, b=rg.b, entry=rg.entry, R=rg.R)

    def sources(self, near: np.ndarray) -> np.ndarray:
        """The regions a home is freed through, from its near cells (flat): the component of
        each one that can take demand at baseline (open, and not split), and for the others the
        pieces lifted.Regions.R lists there (a closed cell's covering pieces; a split cell's,
        those whose clearing alone makes its parts one)."""
        sub = self.R[near].tocsr()
        has = sub[:, :self.ncomp].getnnz(axis=1) > 0
        return np.unique(np.concatenate([sub[has][:, :self.ncomp].indices, sub[~has].indices]))


class Crossing(NamedTuple):
    """A home's cheapest crossing in a round of free_first."""
    home: int
    cleared: list[int]          # the unheld owners of its path's pieces, ascending
    D: float                    # their population share (Clearing.cost)


def free_first(c: Clearing, *, budget: float, held: np.ndarray) -> np.ndarray:
    """B's first stage (owner 2026-10-11, "B SGTM"): `held` plus crossings that free the homes
    enclosed at baseline, the least D per unit of the home's population first, while the held
    set's D (c.cost) stays within `budget`. Each round one Dijkstra from the street over the
    RegionGraph, entering a piece costing its owner's share (a component, or a piece of a held
    building, 0; FREE_STEP more a step); a home's crossing is the unheld owners of the pieces on
    the cheapest path to one of its source regions (RegionGraph.sources of Placement.near), and
    once it is held the home must be freed by the metric's own test (Placement.reached). A path
    through two pieces of one building pays that building twice in the search, though the
    crossing's D is the true sum: conservative, so the graph claims no path the raster lacks. A
    crossing may clear the home's own building (one with footprint all round): the metric counts
    it freed, as it keeps every cleared home's residents. No home enclosed: `held` at once, no
    geometry (any mesh)."""
    sc = c.sc
    if not sc.enclosed.any():
        return held.copy()
    g, bid = sc.grid, sc.block.block_id
    rg = RegionGraph.of(c)          # on any mesh
    street = rg.ncomp + len(rg.owner)                       # the street's node, after them
    a = np.concatenate([rg.a, np.full(len(rg.entry), street)])
    b = np.concatenate([rg.b, rg.entry])
    near = sc.pl.near.tocsc()
    src = {int(h): rg.sources(near.indices[near.indptr[h]:near.indptr[h + 1]])
           for h in np.flatnonzero(sc.enclosed)}
    held = held.copy()
    left = sc.enclosed & ~common.reached(sc.pl, g, c.open_(held), c.p)
    spent, k = 0.0, 0
    while left.any():
        enter = np.concatenate([np.zeros(rg.ncomp),
                                np.where(held[rg.owner], 0.0, c.cost[rg.owner])])
        G = sp.csr_matrix((enter[b] + FREE_STEP, (a, b)), shape=(street + 1, street + 1))
        dist, pred, _ = dijkstra(G, directed=True, indices=[street], min_only=True,
                                 return_predecessors=True)
        room = budget - float(c.cost[held].sum())
        best: Crossing | None = None
        key = np.inf                                        # best's D per unit of population
        for h in np.flatnonzero(left):
            s = src[int(h)]
            if not np.isfinite(dist[s]).any():      # a part of the block no street reaches
                continue
            t, path = int(s[np.argmin(dist[s])]), []
            while t != street:
                path.append(t)
                t = int(pred[t])
            own = {int(rg.owner[r - rg.ncomp]) for r in path if r >= rg.ncomp}
            new = sorted(j for j in own if not held[j])
            D = float(c.cost[new].sum())
            if D <= room and D / sc.w[h] < key:
                best, key = Crossing(home=int(h), cleared=new, D=D), D / sc.w[h]
        if best is None:
            break
        held[best.cleared] = True
        freed = common.reached(sc.pl, g, c.open_(held), c.p)
        if not freed[best.home]:
            raise RuntimeError(f"{bid}: home {best.home}'s crossing {best.cleared} does not free "
                               "it: the region graph claimed a path the raster does not have")
        spent, k = spent + best.D, k + 1
        also = [int(j) for j in np.flatnonzero(left & freed) if j != best.home]
        print(f"{bid} free: home {best.home} ({c.cost[best.home]:.4f} of the population) D "
              f"{best.D:.4f}, clearing {best.cleared}, own building "
              f"{'yes' if best.home in best.cleared else 'no'}"
              + (f", freeing {also} too" if also else ""), flush=True)
        left &= ~freed
    n0 = int(sc.enclosed.sum())
    print(f"{bid} free: {n0 - int(left.sum())} of {n0} enclosed homes freed, D {spent:.4f} on "
          f"{k} crossings, {int(left.sum())} left enclosed", flush=True)
    return held


class GramSource(Protocol):
    """How alike the candidates' effects are: a PSD matrix over `cand` whose normalised entries
    are the correlations rho_ij that `search.Diverse` builds its batch from."""
    @property
    def name(self) -> str: ...

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
    home's weight in J_power (inj_h u_h^power, inj the demand's injection). The sweep runs where
    `sweep` says."""
    sweep: Sweep

    @property
    def name(self) -> str:
        return "cat"

    def gram(self, c: Clearing, t: Tension, cand: list[int], power: float) -> np.ndarray:
        V = self.shares(c, t, cand)
        wt = t.system.xp.asarray(c.sc.inj * np.nan_to_num(t.home_u) ** power)
        return t.solver.to_host((V * wt[:, None]).T @ V)

    def shares(self, c: Clearing, t: Tension, cand: list[int]):
        """v_j(h) on the device, (homes x candidates): the share of home h's injected flow in
        the tension's state that passes through candidate j's cells on its way to ground (0 for
        a home none of whose injection is on a kept unknown)."""
        sy, K, xp = t.system, t.K, t.system.xp
        spm = t.solver.sparse
        A = sy.A
        keep = xp.flatnonzero(sy.keep)
        u = t.u[keep]
        rowsum = A @ xp.ones(A.shape[0])
        assert len(sy.unk_cell) == c.bfree.shape[1]     # eps world: free cell = inside
        pos = -xp.ones(sy.n, dtype=xp.int64)
        pos[keep] = xp.arange(len(keep))
        # per home, its injection on each kept unknown (sy.load's: S_cj inj_j by the axis's
        # angular share), so a cell several homes inject into splits its load among them
        Sc = c.sc.pl.S.tocoo()
        unk = sy.unk_cell[sy.cell.ravel()[xp.asarray(Sc.row)]]         # -1 on ground
        on = unk >= 0
        at = pos[(unk[on][:, None] * K + xp.arange(K)[None, :]).ravel()]
        home = xp.repeat(xp.asarray(Sc.col)[on], K)
        load = (xp.asarray(Sc.data * c.sc.inj[Sc.col])[on][:, None]
                * xp.asarray(lifted.axes(K)[2] / np.pi)[None, :]).ravel()
        kept = at >= 0
        inj = xp.bincount(home[kept], load[kept], minlength=c.n)
        Hown = spm.coo_matrix((load[kept], (home[kept], at[kept])),
                              shape=(c.n, len(keep))).tocsr()
        # members: kept positions of each candidate's unknowns
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
        return xp.where(inj[:, None] > 0, V / xp.where(inj > 0, inj, 1.0)[:, None], 0.0)


def _spread(gain: np.ndarray, cost: np.ndarray, H: np.ndarray, need: float,
            first: int | None = None) -> list[int]:
    """Greedy batch under a normalised quadratic model: with rho = H's correlations, candidate i
    is worth g_i - sum_{j in batch} rho_ij sqrt(g_i g_j) once the batch is chosen (a duplicate of
    a chosen one is worth 0, an independent one keeps its g_i, a complementary one, rho < 0,
    gains). Take the best worth per unit cost until the batch's cost reaches `need`; `first`, if
    given, is taken first whatever its worth (Diverse's screening width)."""
    d = np.sqrt(np.clip(np.diag(H), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(np.outer(d, d) > 0, H / np.outer(d, d), 0.0)
    sg = np.sqrt(np.clip(gain, 0.0, None))
    worth = gain.astype(float).copy()
    avail = np.ones(len(gain), dtype=bool)
    chosen: list[int] = []
    taken = 0.0
    while taken < need - 1e-12 and avail.any():
        i = (first if first is not None and not chosen
             else int(np.argmax(np.where(avail, worth / cost, -np.inf))))
        chosen.append(i)
        avail[i] = False
        taken += float(cost[i])
        worth -= rho[:, i] * sg * sg[i]
    return chosen


def greedy_block(b, rnd: engine.Round, h: float, d_max: float, out: Path, population,
                 power: float, cond: lifted.Conductance, solver: lifted.Solver,
                 search: lifted.AlongConductance | None) -> None:
    """The greedy to d_max under the conductance `cond`. Records perm (in J_power) and perm1
    (the p = 1 score) at every step; `cleared` lists the step's buildings in pick order."""
    mesh = lifted.UniformMesh(h, offset=lifted.OFFSET)
    c = Clearing(b, mesh, lifted.Params(ell_m=3.0, K=8, along=cond.along, solver=solver,
                                        cells=cond.cells),
                 population=population, search=search, demand=common.PopulationDemand())
    sc = c.sc
    J0 = sc.J(sc.u0, power)
    P0 = sc.P0
    import search as engine     # search imports clear, so here; `search` is the conductance
    t0 = time.time()
    rec = engine.GreedyRows(block_id=b.block_id, c=c, power=power, J0=J0, P0=P0, t0=t0)
    engine.greedy(rnd, d_max)(engine.SearchState.start(
        c, power=power, score=engine.Score(J0, P0), world=engine.EXACT), rec)
    rows = rec.rows
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
                     common.POPULATIONS[_CFG["pop"]], _CFG["power"], _CFG["cond"], _CFG["solver"],
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


def rows_dir(picker: str, h: float, pop: str, power: float, along: str) -> Path:
    """The rows of a greedy-format run, `along` its conductance's name (lifted.conductance_of's,
    [@<search along>]): a run before the face aperture (2026-10-13) named its along-conductance
    alone, `uni` by leaving it out."""
    return HERE / (f"clear_rows_{picker}_h{h:g}" + ("" if pop == "count" else f"_{pop}")
                   + ("" if power == 1.0 else f"_p{power:g}")
                   + ("" if along == "uni" else f"_{along}"))


def run(workers: int, picker: engine.Round, h: float, d_max: float, pop: str, power: float,
        cond: lifted.Conductance, solver: lifted.Solver,
        search: lifted.AlongConductance | None, ids: list[str] | None) -> None:
    """All 220 study blocks, largest first, or just `ids`. `search`: the tension's
    along-conductance when it differs from the scoring one (rows dir `<cond>@<search>`)."""
    import multiprocessing
    global _BLOCKS, _CFG
    d = rows_dir(picker.name, h, pop, power,
                 cond.name + ("" if search is None else f"@{search.name}"))
    d.mkdir(exist_ok=True)
    _CFG = dict(picker=picker, h=h, d_max=d_max, dir=d, pop=pop, power=power, cond=cond,
                solver=solver, search=search)
    _BLOCKS = common.build_blocks(common.recipients() if ids is None else ids)
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        failed = [f for f in pool.imap_unordered(_one, list(range(len(_BLOCKS)))[::-1]) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


if __name__ == "__main__":
    import clear        # run as a script this file would be `__main__`, and search's `clear` a
                        # second copy of this module (its own EXACT-side classes and helpers)
    import search as engine     # `search` is a conductance elsewhere in this module
    if sys.argv[1] == "loo":
        clear.loo(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
                  sys.argv[4] if len(sys.argv) > 4 else "count",
                  float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
    elif sys.argv[1] in ("run", "some"):
        # run <workers> <picker> <h> <d_max> <pop> <p> <conductance>[@<search along>] <solver>
        # [ids,]
        scans = lifted.scans_of(sys.argv[9])
        specs = sys.argv[8].split("@")
        clear.run(int(sys.argv[2]), engine.greedy_round(sys.argv[3], clear.sweep_of(sys.argv[9])),
                  float(sys.argv[4]), float(sys.argv[5]),
                  sys.argv[6], float(sys.argv[7]), lifted.conductance_of(specs[0], scans),
                  lifted.solver_of(sys.argv[9]),
                  lifted.along_of(specs[1], scans) if len(specs) == 2 else None,
                  sys.argv[10].split(",") if sys.argv[1] == "some" else None)
