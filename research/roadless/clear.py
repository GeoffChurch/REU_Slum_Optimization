"""Road-free clearing: remove whole buildings, greedily, by the roadless score.

Tension. Solve once with the remaining buildings as a poor conductor (open fraction eps
wherever a footprint is), so the potential is defined inside them. dP/dw_e = -(u_a - u_b)^2, so
a building's first-order gain from clearing is the sum over the edges it covers of
(Delta u)^2 x (the weight gained). That is the TENSION: how hard the escape flow presses on it.

Screened greedy. Each step: one tension solve, exact solves for the top M buildings by tension,
clear the best. `loo` checks the screen against exact leave-one-out on one step.

    PYTHONPATH=. pixi run python research/roadless/clear.py loo <block idx> [h]
    PYTHONPATH=. pixi run python research/roadless/clear.py run <workers> <M> <h> <d_max> [pop] [p]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

EPS = 0.01


class Clearing:
    def __init__(self, block, h: float, p: lifted.Params, population=None):
        self.sc = common.Scorer(block, h, p, population=population)
        self.cost = self.sc.w / self.sc.w.sum()        # population share of each building
        self.p = p
        g = self.sc.grid
        # sub-sample -> building index (-1 = none); overlaps resolved arbitrarily
        X, Y = g._sub_xy()
        lab = -np.ones(g.isub.shape, dtype=np.int64)
        pts_idx = np.nonzero(g.bsub)
        pts = shapely.points(X[pts_idx], Y[pts_idx])
        hit_pt, hit_poly = self.sc.tree.query(pts, predicate="within")
        lab_flat = -np.ones(len(pts), dtype=np.int64)
        lab_flat[hit_pt] = hit_poly
        lab[pts_idx] = lab_flat
        self.lab = lab
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

    def tension(self, power: float = 1.0) -> np.ndarray:
        """First-order gain in J_power of clearing each (remaining) building, from one
        eps-material solve (power 1: the potential is its own adjoint) or two (power != 1: the
        adjoint L lam = dJ/du = power u_i^(power-1) x injection). dJ/dw_e = -(du_e)(dlam_e)."""
        g = self.sc.grid
        op = self.open_(self.removed)
        full = g.isub.mean(axis=-1)                       # everything open
        op_eps = op + EPS * (full - op)
        sol = lifted.solve(g, op_eps, self.sc.f, self.p)
        K = self.p.K
        nc = int((op_eps > 0).sum())
        uk = np.zeros((nc, K))
        live = sol.unk_cell >= 0
        uk[live] = sol.u.reshape(-1, K)
        if power == 1.0:
            lk = uk
        else:
            uh = self.sc.home_u_of(sol, op_eps)
            mult = np.zeros_like(self.sc.f)
            on = self.sc.owner >= 0
            mult[on] = power * np.nan_to_num(uh[self.sc.owner[on]]) ** (power - 1.0)
            lam = lifted.solve(g, op_eps, self.sc.f * mult, self.p)
            lk = np.zeros((nc, K))
            lk[live] = lam.u.reshape(-1, K)
        # per cell: sum over incident edges of (du)^2 x (weight fully open - weight now), halved
        T = np.zeros(nc)
        cur = {k: (a, b, w) for k, a, b, w in lifted.along_edges(op_eps, self.p)}
        for k, a, b, wf in lifted.along_edges(full, self.p):
            ca, cb, wc = cur[k]
            assert len(ca) == len(a) and (ca == a).all()
            gain = (uk[a, k] - uk[b, k]) * (lk[a, k] - lk[b, k]) * (wf - wc)
            np.add.at(T, a, 0.5 * gain)
            np.add.at(T, b, 0.5 * gain)
        _v, _th, _m, gap = lifted.axes(K)
        free_ids = np.arange(nc)
        dvol = (full - op_eps)[op_eps > 0]
        for k in range(K):
            k2 = (k + 1) % K
            T += (uk[free_ids, k] - uk[free_ids, k2]) * (lk[free_ids, k] - lk[free_ids, k2]) * dvol * (
                g.h * g.h / (self.p.ell_m ** 2 * gap[k]))
        # grid cell (row-major over all cells) -> node id
        node = -np.ones(g.inside.size, dtype=np.int64)
        node[np.flatnonzero(op_eps > 0)] = np.arange(nc)
        out = np.zeros(self.n)
        for j, (cells, frac) in self.bcells.items():
            if self.removed[j]:
                continue
            nd = node[cells]
            ok = nd >= 0
            # the cell's tension is shared by the buildings covering it, by covered fraction
            out[j] = float((T[nd[ok]] * frac[ok] / np.maximum(1 - op[np.unravel_index(
                cells[ok], op.shape)], 1e-9)).sum())
        out[self.removed] = -np.inf
        return out


def loo(i: int, h: float, pop: str = "count", power: float = 1.0) -> None:
    from scipy.stats import spearmanr
    blocks = common.build_blocks(common.recipients())
    b = blocks[i]
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8), population=common.POPULATIONS[pop])
    sc = c.sc
    P0 = sc.J(sc.u0, power)
    t = time.time()
    T = c.tension(power)
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


def greedy_block(b, M: int, h: float, d_max: float, out: Path, population,
                 power: float = 1.0) -> None:
    """Each step: rank remaining buildings by tension (in J_power) per unit of population,
    exact-solve the top M, clear the one with the best exact J_power gain per unit of population
    displaced. Records perm (in J_power) and perm1 (the p = 1 score) at every step."""
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8), population=population)
    sc = c.sc
    J0 = sc.J(sc.u0, power)
    P0 = sc.P0
    P = J0
    rows = [dict(block=b.block_id, n=c.n, step=0, D=0.0, perm=0.0, perm1=0.0, cleared=-1,
                 P0=P0)]
    step = 0
    t0 = time.time()
    while c.cost[c.removed].sum() < d_max - 1e-12 and not c.removed.all():
        T = c.tension(power) / c.cost
        cand = [j for j in np.argsort(-T)[:M] if not c.removed[j]]
        best, bestP, bestv = -1, np.inf, -np.inf
        for j in cand:
            r = c.removed.copy()
            r[j] = True
            op = c.open_(r)
            sol = lifted.solve(sc.grid, op, sc.f, c.p)
            Pj = sc.J(sc.home_u_of(sol, op), power)
            v = (P - Pj) / c.cost[j]
            if v > bestv:
                best, bestP, bestv, bestP1 = int(j), Pj, v, sol.P
        c.removed[best] = True
        P = bestP
        step += 1
        rows.append(dict(block=b.block_id, n=c.n, step=step, D=float(c.cost[c.removed].sum()),
                         perm=1 - (bestP / J0) ** (1 / power), perm1=1 - bestP1 / P0,
                         cleared=best, P0=P0))
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
        greedy_block(b, _CFG["M"], _CFG["h"], _CFG["d_max"], out,
                     common.POPULATIONS[_CFG["pop"]], _CFG["power"])
    except Exception as e:
        print(f"{b.block_id} FAILED {type(e).__name__}: {e}"[:300], flush=True)


def rows_dir(M: int, h: float, pop: str, power: float = 1.0) -> Path:
    return HERE / (f"clear_rows_M{M}_h{h:g}" + ("" if pop == "count" else f"_{pop}")
                   + ("" if power == 1.0 else f"_p{power:g}"))


def run(workers: int, M: int, h: float, d_max: float, pop: str, power: float) -> None:
    import multiprocessing
    global _BLOCKS, _CFG
    d = rows_dir(M, h, pop, power)
    d.mkdir(exist_ok=True)
    _CFG = dict(M=M, h=h, d_max=d_max, dir=d, pop=pop, power=power)
    _BLOCKS = common.build_blocks(common.recipients())
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for _ in pool.imap_unordered(_one, list(range(len(_BLOCKS)))[::-1]):
            pass


if __name__ == "__main__":
    if sys.argv[1] == "loo":
        loo(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
            sys.argv[4] if len(sys.argv) > 4 else "count",
            float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
    elif sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
            sys.argv[6] if len(sys.argv) > 6 else "count",
            float(sys.argv[7]) if len(sys.argv) > 7 else 1.0)
