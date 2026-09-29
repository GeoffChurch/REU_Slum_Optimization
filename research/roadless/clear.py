"""Road-free clearing: remove whole buildings, greedily, by the roadless score.

Tension. Solve once with the remaining buildings as a poor conductor (open fraction eps
wherever a footprint is), so the potential is defined inside them. dP/dw_e = -(u_a - u_b)^2, so
a building's first-order gain from clearing is the sum over the edges it covers of
(Delta u)^2 x (the weight gained). That is the TENSION: how hard the escape flow presses on it.

Screened greedy. Each step: one tension solve, exact solves for the top M buildings by tension,
clear the best. `loo` checks the screen against exact leave-one-out on one step.

    PYTHONPATH=. pixi run python research/roadless/clear.py loo <block idx> [h]
    PYTHONPATH=. pixi run python research/roadless/clear.py run <workers> <M> <h> <d_max>
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
    def __init__(self, block, h: float, p: lifted.Params):
        self.sc = common.Scorer(block, h, p)
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

    def tension(self) -> np.ndarray:
        """First-order gain of clearing each (remaining) building, from one eps-material solve."""
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
        # per cell: sum over incident edges of (du)^2 x (weight fully open - weight now), halved
        T = np.zeros(nc)
        cur = {k: (a, b, w) for k, a, b, w in lifted.along_edges(op_eps, self.p)}
        for k, a, b, wf in lifted.along_edges(full, self.p):
            ca, cb, wc = cur[k]
            assert len(ca) == len(a) and (ca == a).all()
            gain = (uk[a, k] - uk[b, k]) ** 2 * (wf - wc)
            np.add.at(T, a, 0.5 * gain)
            np.add.at(T, b, 0.5 * gain)
        _v, _th, _m, gap = lifted.axes(K)
        free_ids = np.arange(nc)
        dvol = (full - op_eps)[op_eps > 0]
        for k in range(K):
            k2 = (k + 1) % K
            T += (uk[free_ids, k] - uk[free_ids, k2]) ** 2 * dvol * (
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


def loo(i: int, h: float) -> None:
    from scipy.stats import spearmanr
    blocks = common.build_blocks(common.recipients())
    b = blocks[i]
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8))
    P0 = c.sc.P0
    t = time.time()
    T = c.tension()
    tt = time.time() - t
    t = time.time()
    exact = np.zeros(c.n)
    for j in range(c.n):
        r = c.removed.copy()
        r[j] = True
        exact[j] = P0 - c.P(r)
    te = time.time() - t
    rho = spearmanr(T, exact).statistic
    top = np.argsort(-exact)
    rankT = np.argsort(np.argsort(-T))
    print(f"{b.block_id} n={c.n} h={h}: tension {tt:.0f}s, exact LOO {te:.0f}s "
          f"({te / c.n:.1f}s each)", flush=True)
    print(f"  Spearman(tension, exact gain) {rho:+.3f}; tension rank of the exact top 5: "
          f"{rankT[top[:5]].tolist()}; exact gain top 5 (share of P0): "
          f"{np.round(exact[top[:5]] / P0, 4).tolist()}", flush=True)
    for M in (1, 4, 8, 16):
        pick = np.argsort(-T)[:M]
        print(f"  M={M:2d}: best exact gain among tension top-M = "
              f"{exact[pick].max() / exact.max():.3f} of the true best", flush=True)


def greedy_block(b, M: int, h: float, d_max: float, out: Path) -> None:
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8))
    P0 = c.sc.P0
    rows = [dict(block=b.block_id, n=c.n, step=0, D=0.0, perm=0.0, cleared=-1, P0=P0)]
    step = 0
    t0 = time.time()
    while c.removed.sum() / c.n < d_max - 1e-12 and not c.removed.all():
        T = c.tension()
        cand = [j for j in np.argsort(-T)[:M] if not c.removed[j]]
        best, bestP = -1, np.inf
        for j in cand:
            r = c.removed.copy()
            r[j] = True
            Pj = c.P(r)
            if Pj < bestP:
                best, bestP = int(j), Pj
        c.removed[best] = True
        step += 1
        rows.append(dict(block=b.block_id, n=c.n, step=step, D=c.removed.sum() / c.n,
                         perm=1 - bestP / P0, cleared=best, P0=P0))
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
        greedy_block(b, _CFG["M"], _CFG["h"], _CFG["d_max"], out)
    except Exception as e:
        print(f"{b.block_id} FAILED {type(e).__name__}: {e}"[:300], flush=True)


def run(workers: int, M: int, h: float, d_max: float) -> None:
    import multiprocessing
    global _BLOCKS, _CFG
    d = HERE / f"clear_rows_M{M}_h{h:g}"
    d.mkdir(exist_ok=True)
    _CFG = dict(M=M, h=h, d_max=d_max, dir=d)
    _BLOCKS = common.build_blocks(common.recipients())
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for _ in pool.imap_unordered(_one, list(range(len(_BLOCKS)))[::-1]):
            pass


if __name__ == "__main__":
    if sys.argv[1] == "loo":
        loo(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5)
    elif sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
