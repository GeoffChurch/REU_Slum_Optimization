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
    PYTHONPATH=. pixi run python research/roadless/clear.py run <workers> <picker> <h> <d_max> <pop> <p> <along>
    PYTHONPATH=. pixi run python research/roadless/clear.py some <workers> <picker> <h> <d_max> <pop> <p> <along> <ids,>

picker: M4 | B0.01g3 | S0.01cat; along: uni | sl<beta> | ss<beta>k<kappa>
"""
from __future__ import annotations

import dataclasses
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

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


def _cells_axes(sy: lifted.System, x: np.ndarray, K: int) -> np.ndarray:
    """Per-unknown vector -> (free cells, K), zero on ground."""
    out = np.zeros((len(sy.unk_cell), K))
    live = sy.unk_cell >= 0
    out[live] = x.reshape(-1, K)
    return out


class Tension:
    """One eps-material solve's worth of state: the factorized system, primal u and adjoint lam
    per unknown, the edges (ca, cb, ka, kb, dw, w) with their weight now and the weight clearing
    everything would add at the current along-factors, and A (free cells x buildings), the share
    of each cell's gain a building owns. g = the LOCAL first-order gain (a building's own edges,
    beta_e = (A[a] + A[b]) / 2 of each) + the NONLOCAL gain (the along-factors' response
    elsewhere, e.g. straight runs lengthening through the cleared cells)."""

    def __init__(self, system: lifted.System, u, lam, fam, A, K: int, home_u, removed,
                 nonlocal_gain):
        self.system, self.u, self.lam, self.fam, self.A, self.K = system, u, lam, fam, A, K
        self.home_u = home_u                  # per building, from the eps solve
        self.g_local = self._gain()
        self.g = self.g_local + nonlocal_gain
        self.g[removed] = -np.inf

    def _gain(self) -> np.ndarray:
        uk = _cells_axes(self.system, self.u, self.K)
        lk = _cells_axes(self.system, self.lam, self.K)
        nc = uk.shape[0]
        T = np.zeros(nc)
        for ca, cb, ka, kb, dw, _w in self.fam:
            q = dw * (uk[ca, ka] - uk[cb, kb]) * (lk[ca, ka] - lk[cb, kb])
            T += 0.5 * (np.bincount(ca, q, minlength=nc) + np.bincount(cb, q, minlength=nc))
        return np.asarray(self.A.T @ T)

class Clearing:
    def __init__(self, block, h: float, p: lifted.Params, population=None,
                 rtol: float = RTOL_TENSION):
        self.sc = common.Scorer(block, h, p, population=population)
        self.cost = self.sc.w / self.sc.w.sum()        # population share of each building
        self.p = p
        self.rtol = rtol                               # for the tension solves
        g = self.sc.grid
        # sub-sample -> building index (-1 = none); overlaps go to the later footprint
        self.lab = lab = g.label_sub(self.sc.polys)
        self.n = len(self.sc.polys)
        # per building: its cells and the fraction of each it covers
        ny, nx, S2 = lab.shape
        flat = lab.reshape(-1, S2)
        cells, subs = np.nonzero(flat >= 0)
        b = flat[cells, subs]
        df = pd.DataFrame(dict(b=b, c=cells)).value_counts().reset_index(name="k")
        self.bcells = {int(k): (g2.c.to_numpy(), g2.k.to_numpy() / S2)
                       for k, g2 in df.groupby("b")}
        self.removed = np.zeros(self.n, dtype=bool)

    def open_(self, removed: np.ndarray) -> np.ndarray:
        g = self.sc.grid
        cleared = np.isin(self.lab, np.nonzero(removed)[0])
        return (g.isub & (~g.bsub | cleared)).mean(axis=-1)

    def P(self, removed: np.ndarray) -> float:
        return self.sc.P_free(self.open_(removed))

    def tension(self, power: float = 1.0) -> Tension:
        """First-order gain in J_power of clearing each (remaining) building, from one
        eps-material solve (power 1: the potential is its own adjoint) or two (power != 1: the
        adjoint L lam = dJ/du = power u_i^(power-1) x injection). dJ/dw_e = -(du_e)(dlam_e)."""
        g = self.sc.grid
        op = self.open_(self.removed)
        full = g.isub.mean(axis=-1)                       # everything open
        op_eps = op + EPS * (full - op)
        sy = lifted.System(g, op_eps, self.p)
        sol = lifted.solve(g, op_eps, self.sc.f, self.p, system=sy, rtol=self.rtol)
        if power == 1.0:
            lam = sol.u
        else:
            uh = self.sc.home_u_of(sol, op_eps)
            mult = np.zeros_like(self.sc.f)
            on = self.sc.owner >= 0
            mult[on] = power * np.nan_to_num(uh[self.sc.owner[on]]) ** (power - 1.0)
            lam = lifted.solve(g, op_eps, self.sc.f * mult, self.p, system=sy, rtol=self.rtol).u
        # Each edge's weight gained by clearing everything, at the CURRENT along-factors (the
        # local term): w = w_uniform x min(factor at the two ends). The factors' own response to
        # clearing (straight runs lengthening through the cleared cells, boosting edges
        # elsewhere) is the nonlocal term, from the along-conductance's vjp below.
        K = self.p.K
        pu = dataclasses.replace(self.p, along=lifted.Uniform())
        Fl = self.p.along.layers(op_eps, g.h, K)
        rr, cc = np.nonzero(op_eps > 0)
        uk = _cells_axes(sy, sol.u, K)
        lk = _cells_axes(sy, lam, K)
        Agrad = np.zeros((K, *op_eps.shape))
        fam = []
        for i, ((ca, cb, ka, kb, wf), (ca2, cb2, _, _, wc)) in enumerate(zip(
                lifted.edges(full, g.h, pu), lifted.edges(op_eps, g.h, pu), strict=True)):
            assert (ca == ca2).all() and (cb == cb2).all()
            if i < K:                                   # along family, layer i
                Fi = np.broadcast_to(Fl[i], op_eps.shape)
                fa, fb = Fi[rr[ca], cc[ca]], Fi[rr[cb], cc[cb]]
                boost = np.minimum(fa, fb)
                q = (uk[ca, ka] - uk[cb, kb]) * (lk[ca, ka] - lk[cb, kb])    # -dJ/dw_e
                at_a = fa <= fb
                np.add.at(Agrad[i], (np.where(at_a, rr[ca], rr[cb]),
                                     np.where(at_a, cc[ca], cc[cb])), q * wc)
                fam.append((ca, cb, ka, kb, (wf - wc) * boost, wc * boost))
            else:
                fam.append((ca, cb, ka, kb, wf - wc, wc))
        # nonlocal gain: d(gain)/d(open) at every cell, x the open each building would add
        dgain = self.p.along.vjp(op_eps, g.h, K, Agrad).ravel()
        nonlocal_gain = np.zeros(self.n)
        for j, (cells, frac) in self.bcells.items():
            nonlocal_gain[j] = float((dgain[cells] * frac).sum()) * (1.0 - EPS)
        # attribution: building j owns frac/(1 - op) of each cell it covers
        node = -np.ones(g.inside.size, dtype=np.int64)
        node[np.flatnonzero(op_eps > 0)] = np.arange(int((op_eps > 0).sum()))
        opf = op.ravel()
        r, c, v = [], [], []
        for j, (cells, frac) in self.bcells.items():
            if self.removed[j]:
                continue
            nd = node[cells]
            ok = nd >= 0
            r.append(nd[ok]); c.append(np.full(int(ok.sum()), j))
            v.append(frac[ok] / np.maximum(1 - opf[cells[ok]], 1e-9))
        A = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                          shape=(len(node[node >= 0]), self.n))
        return Tension(system=sy, u=sol.u, lam=lam, fam=fam, A=A, K=self.p.K,
                       home_u=self.sc.home_u_of(sol, op_eps), removed=self.removed,
                       nonlocal_gain=nonlocal_gain)

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
class Catchment:
    """No solves: trace the round's flow downhill and give each candidate the share of each
    home's flow that passes through its cells. H_ij = sum_h c_h v_i(h) v_j(h), with c_h the
    home's weight in J_power (w_h u_h^power)."""
    chunk: int = 16

    @property
    def name(self) -> str:
        return "cat"

    def gram(self, c: Clearing, t: Tension, cand: list[int], power: float) -> np.ndarray:
        sy, K = t.system, t.K
        A = sy.A
        keep = np.flatnonzero(sy.keep)
        u = t.u[keep]
        rowsum = np.asarray(A.sum(axis=1)).ravel()
        order = np.argsort(u, kind="stable")
        # per kept unknown: its home (the owner of its cell) and its injection
        free = sy.cell >= 0
        owner_free = c.sc.owner[free]
        live = sy.unk_cell >= 0
        owner_unk = np.repeat(owner_free[live], K)[keep]
        b = sy.load(c.sc.f, c.p)[keep]
        home = owner_unk >= 0
        inj = np.bincount(owner_unk[home], b[home], minlength=c.n)
        cellA = t.A[:, cand].tocsc()
        cell_unk = sy.unk_cell
        pos = -np.ones(sy.n, dtype=np.int64)
        pos[keep] = np.arange(len(keep))
        V = np.zeros((c.n, len(cand)))
        for s0 in range(0, len(cand), self.chunk):
            cols = range(s0, min(s0 + self.chunk, len(cand)))
            inA = np.zeros((len(keep), len(cols)), dtype=np.bool_)
            for m, i in enumerate(cols):
                cells = cellA[:, i].indices
                un = cell_unk[cells]
                un = un[un >= 0]
                idx = pos[(un[:, None] * K + np.arange(K)[None, :]).ravel()]
                inA[idx[idx >= 0], m] = True
            out = np.zeros((len(keep), len(cols)))
            _catch(A.indptr, A.indices, A.data, u, rowsum, order, inA, out)
            for m, i in enumerate(cols):
                V[:, i] = np.bincount(owner_unk[home], b[home] * out[home, m], minlength=c.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            V = np.where(inj[:, None] > 0, V / inj[:, None], 0.0)
        wt = c.sc.w * np.nan_to_num(t.home_u) ** power
        return (V * wt[:, None]).T @ V


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
        return Picked(taken, *_exact(c, taken, power))


def greedy_block(b, picker: Picker, h: float, d_max: float, out: Path, population,
                 power: float, along: lifted.AlongConductance) -> None:
    """Each step: rank remaining buildings by tension (in J_power) per unit of population and
    let `picker` clear some. Records perm (in J_power) and perm1 (the p = 1 score) at every
    step; `cleared` lists the step's buildings in pick order."""
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8, along=along), population=population)
    sc = c.sc
    J0 = sc.J(sc.u0, power)
    P0 = sc.P0
    J = J0
    rows = [dict(block=b.block_id, n=c.n, step=0, D=0.0, perm=0.0, perm1=0.0, cleared=[],
                 P0=P0, t=0.0)]
    step = 0
    t0 = time.time()
    while c.cost[c.removed].sum() < d_max - 1e-12 and not c.removed.all():
        pk = picker.pick(c, c.tension(power), J, power)
        c.removed[pk.cleared] = True
        J = pk.J
        step += 1
        rows.append(dict(block=b.block_id, n=c.n, step=step, D=float(c.cost[c.removed].sum()),
                         perm=1 - (pk.J / J0) ** (1 / power), perm1=1 - pk.P1 / P0,
                         cleared=pk.cleared, P0=P0, t=time.time() - t0))
        if c.n > 1000:
            print(f"  {b.block_id} step {step} D {rows[-1]['D']:.3f} perm' {rows[-1]['perm']:.3f}"
                  f" {rows[-1]['t']:.0f}s", flush=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, out)
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={c.n} {step} steps, perm' at end "
          f"{rows[-1]['perm']:.3f}  {time.time() - t0:.0f}s", flush=True)


_BLOCKS: list = []
_CFG: dict = {}


def _one(i: int) -> None:
    b = _BLOCKS[i]
    out = _CFG["dir"] / f"{b.block_id}.parquet"
    if out.exists():
        return
    try:
        greedy_block(b, _CFG["picker"], _CFG["h"], _CFG["d_max"], out,
                     common.POPULATIONS[_CFG["pop"]], _CFG["power"], _CFG["along"])
    except Exception as e:
        print(f"{b.block_id} FAILED {type(e).__name__}: {e}"[:300], flush=True)


def cleared_through(g: pd.DataFrame, step: int) -> np.ndarray:
    """Buildings cleared in steps 1..step of one block's greedy rows, in clearing order."""
    g = g[(g.step >= 1) & (g.step <= step)].sort_values("step")
    return np.array([j for js in g.cleared for j in js], dtype=np.int64)


def picker_of(spec: str) -> Picker:
    """`M4` -> Screened(4); `B0.01g3` -> Batched(delta 0.01, gap 3 m); `S0.01cat` ->
    Spread(delta 0.01) with Catchment. (Impact, Sketch and the no-spacing null were measured and
    dominated: NOTES.md, "Batching".)"""
    if spec.startswith("M"):
        return Screened(int(spec[1:]))
    if spec.startswith("B"):
        delta, gap = spec[1:].split("g")
        return Batched(float(delta), float(gap))
    m = re.fullmatch(r"S([0-9.]+)cat", spec)
    if m:
        return Spread(float(m.group(1)), Catchment())
    raise ValueError(f"unknown picker {spec!r}")


def rows_dir(picker: str, h: float, pop: str, power: float = 1.0, along: str = "uni") -> Path:
    return HERE / (f"clear_rows_{picker}_h{h:g}" + ("" if pop == "count" else f"_{pop}")
                   + ("" if power == 1.0 else f"_p{power:g}")
                   + ("" if along == "uni" else f"_{along}"))


def run(workers: int, picker: Picker, h: float, d_max: float, pop: str, power: float,
        along: lifted.AlongConductance, ids: list[str] | None = None) -> None:
    """All 220 study blocks, largest first, or just `ids`."""
    import multiprocessing
    global _BLOCKS, _CFG
    d = rows_dir(picker.name, h, pop, power, along.name)
    d.mkdir(exist_ok=True)
    _CFG = dict(picker=picker, h=h, d_max=d_max, dir=d, pop=pop, power=power, along=along)
    _BLOCKS = common.build_blocks(common.recipients() if ids is None else ids)
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for _ in pool.imap_unordered(_one, list(range(len(_BLOCKS)))[::-1]):
            pass


if __name__ == "__main__":
    if sys.argv[1] == "loo":
        loo(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
            sys.argv[4] if len(sys.argv) > 4 else "count",
            float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
    elif sys.argv[1] in ("run", "some"):
        # run <workers> <picker> <h> <d_max> <pop> <p> <along> [ids,]
        run(int(sys.argv[2]), picker_of(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
            sys.argv[6], float(sys.argv[7]), lifted.along_of(sys.argv[8]),
            sys.argv[9].split(",") if sys.argv[1] == "some" else None)
