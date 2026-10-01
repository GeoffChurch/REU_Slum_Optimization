"""The clearing problem as one convex program, solved by Frank-Wolfe.

Clear a fraction x_j in [0, 1] of each building: a cell's open fraction is ff0 + sum_j x_j
(the share of the cell building j covers), so it is affine in x. Each edge weight is a minimum
of open fractions (concave in x), and by the Dirichlet principle P(w) = max_u 2 f^T u -
sum_e w_e (du_e)^2 is a maximum of functions affine and non-increasing in w. So P(x) is convex,
and so is min P(x) s.t. sum_j c_j x_j <= D, 0 <= x <= 1 (c_j the population share). J_p for
p != 1 is not convex; Frank-Wolfe is then a local method and the gap is not a bound.

Objective: J_p in the eps world (buildings as eps material, as the tension), so J_eps <= J
for every clearing and a lower bound on J_eps bounds every real clearing. Gradient: dJ/dw_e =
-du_e dlam_e (lam the adjoint; u for p 1); where w_e is a min over several cells, the
gradient is shared equally among the cells attaining it (a supergradient of the min, so a
subgradient of J: the gap stays a valid bound). Linear step: a fractional knapsack (the batch
pickers' rule, best gain per unit population up to D). Step size: a parabola through J(0),
J'(0) = -gap and J(1), checked. Rounding: clear buildings by decreasing x while they fit,
then score exactly.

    CUDA_PATH=/usr PYTHONPATH=. pixi run python research/roadless/relax.py <ids,|all> <p> <cpu|gpu> [iters] [workers]
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import EPS, RTOL_SCORE, Clearing, _cells_axes  # noqa: E402

D_LENS = 0.10
OUT = HERE / "relax_rows"


class Relaxation:
    """One block's relaxation: open fractions, objective and gradient in x. Building j adds
    x_j^q of its cells' open fraction: q 1 is the convex relaxation, q > 1 (SIMP) makes partial
    clearing uneconomic (its conductance falls faster than its cost)."""

    def __init__(self, c: Clearing, power: float, q: float = 1.0):
        self.c, self.power, self.q = c, power, q
        g = c.sc.grid
        self.ff0 = g.ff0.ravel()
        self.inside = np.flatnonzero(g.inside)
        self.J0 = c.sc.J(c.sc.u0, power)

    def open_of(self, x: np.ndarray) -> np.ndarray:
        o = self.ff0.copy()
        o[self.inside] += self.c.bf.T @ (x ** self.q)
        return np.minimum(o, self.c.full.ravel()).reshape(self.c.full.shape)

    def _solve(self, x: np.ndarray):
        c = self.c
        op = self.open_of(x)
        op_eps = op + EPS * (c.full - op)
        sy = lifted.System(c.sc.grid, op_eps, c.p)
        sol = lifted.solve(c.sc.grid, op_eps, c.sc.f, c.p, system=sy, rtol=RTOL_SCORE,
                           host=False)
        th = c.p.solver.to_host
        hsol = dataclasses.replace(sol, u=th(sol.u), cell=th(sol.cell),
                                   unk_cell=th(sol.unk_cell))
        home_u = c.sc.home_u_of(hsol, op_eps)
        return op_eps, sy, sol, home_u

    def value(self, x: np.ndarray) -> float:
        return self.c.sc.J(self._solve(x)[3], self.power)

    def value_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        c, p = self.c, self.c.p
        xp = p.solver.xp
        g = c.sc.grid
        op_eps, sy, sol, home_u = self._solve(x)
        J = c.sc.J(home_u, self.power)
        if self.power == 1.0:
            lam = sol.u
        else:
            mult = np.zeros_like(c.sc.f)
            on = c.sc.owner >= 0
            mult[on] = self.power * np.nan_to_num(home_u[c.sc.owner[on]]) ** (self.power - 1.0)
            lam = lifted.solve(g, op_eps, c.sc.f * mult, p, system=sy, rtol=RTOL_SCORE,
                               host=False).u
        K = p.K
        uk, lk = _cells_axes(sy, sol.u, K), _cells_axes(sy, lam, K)
        o = xp.asarray(op_eps).ravel()
        cell, _nf = lifted._free_ids(o, xp)
        pat = lifted.pattern(g, K, xp)
        v, _th, m, gap = lifted.axes(K)
        gc = xp.zeros(o.size)                           # dJ / d(open) per grid cell
        for k in range(K):
            a, b, line = pat.along[k]
            cells = [a, b, *list(line)]
            vals = xp.stack([o[q] for q in cells])
            act = vals <= vals.min(axis=0)[None, :]      # the cells attaining the min
            ca, cb = cell[a], cell[b]
            q = (uk[ca, k] - uk[cb, k]) * (lk[ca, k] - lk[cb, k])
            dx, dy = int(v[k, 0]), int(v[k, 1])
            coef = -q * (m[k] / float(dx * dx + dy * dy)) / act.sum(axis=0)
            for i, qc in enumerate(cells):
                gc += xp.bincount(qc[act[i]], coef[act[i]], minlength=o.size)
        free = xp.flatnonzero(o > 0)
        for k in range(K):
            k2 = (k + 1) % K
            q = (uk[:, k] - uk[:, k2]) * (lk[:, k] - lk[:, k2])
            gc[free] += -q * (g.h * g.h / (p.ell_m ** 2 * gap[k]))
        grad = (1.0 - EPS) * (c.bf @ p.solver.to_host(gc)[self.inside])
        if self.q != 1.0:
            grad = grad * self.q * x ** (self.q - 1.0)
        return J, grad

    def exact(self, x: np.ndarray) -> float:
        """J_p of a 0/1 clearing in the real geometry (buildings closed)."""
        op = self.open_of(x)
        sol = lifted.solve(self.c.sc.grid, op, self.c.sc.f, self.c.p, rtol=RTOL_SCORE)
        return self.c.sc.J(self.c.sc.home_u_of(sol, op), self.power)

    def perm(self, J: float) -> float:
        return 1.0 - (J / self.J0) ** (1.0 / self.power)


def knapsack(grad: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """argmin <grad, s> s.t. cost . s <= budget, 0 <= s <= 1: the most negative gradient per
    unit cost first, the last one fractional."""
    s = np.zeros(len(grad))
    left = budget
    for j in np.argsort(grad / cost):
        if grad[j] >= 0 or left <= 0:
            break
        s[j] = min(1.0, left / cost[j])
        left -= s[j] * cost[j]
    return s


def round_by_x(x: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """Clear buildings by decreasing x while they fit the budget."""
    r = np.zeros(len(x))
    left = budget + 1e-12
    for j in np.argsort(-x, kind="stable"):
        if x[j] <= 1e-9:
            break
        if cost[j] <= left:
            r[j] = 1.0
            left -= cost[j]
    return r


def frank_wolfe(rel: Relaxation, budget: float, iters: int, log=print) -> dict:
    cost = rel.c.cost
    x = np.zeros(len(cost))
    J, grad = rel.value_grad(x)
    lb, hist = -np.inf, []
    t0 = time.time()
    for it in range(iters):
        s = knapsack(grad, cost, budget)
        d = s - x
        slope = float(grad @ d)                          # J'(0) = -gap
        gap = -slope
        lb = max(lb, J - gap)
        hist.append(dict(it=it, J=J, gap=gap, lb=lb, t=time.time() - t0))
        log(f"    it {it:2d} J {J:.6g} gap {gap:.3g} ({gap / J:.2%}) lb {lb:.6g} "
            f"frac {np.mean((x > 1e-6) & (x < 1 - 1e-6)):.3f} {time.time() - t0:.0f}s")
        if gap <= 1e-4 * J:
            break
        J1 = rel.value(x + d)
        curv = J1 - J - slope
        gam = 1.0 if curv <= 0 else float(np.clip(-slope / (2 * curv), 0.0, 1.0))
        if gam < 1.0:
            Jg = rel.value(x + gam * d)
            if Jg > J1:
                gam, Jg = 1.0, J1
        else:
            Jg = J1
        if Jg >= J:                                     # no progress along d: stop
            break
        x = x + gam * d
        J, grad = rel.value_grad(x)
    return dict(x=x, J=J, lb=lb, hist=hist)


XMIN = 1e-3
QS = (1.5, 2.0, 2.5, 3.0)


def simp(rel: Relaxation, x0: np.ndarray, budget: float, iters: int, log=print,
         move: float = 0.2) -> np.ndarray:
    """SIMP continuation from x0: for q in QS, `iters` optimality-criteria updates x <- clip(
    x (-g / (lam c))^1/2) within `move` of x and [XMIN, 1], lam bisected so that c . x =
    budget. Non-convex: a local method."""
    cost = rel.c.cost
    x = np.clip(x0, XMIN, 1.0)
    x *= min(1.0, budget / float(cost @ x))
    t0 = time.time()
    for q in QS:
        rel.q = q
        for it in range(iters):
            J, g = rel.value_grad(x)
            B = np.maximum(-g, 1e-300) / cost
            lo, hi = 1e-300, 1e300
            for _ in range(200):
                lam = np.sqrt(lo * hi)
                xn = np.clip(x * np.sqrt(B / lam), np.maximum(XMIN, x - move),
                             np.minimum(1.0, x + move))
                lo, hi = (lam, hi) if cost @ xn > budget else (lo, lam)
                if hi / lo < 1 + 1e-9:
                    break
            change = float(np.abs(xn - x).max())
            x = xn
            if it % 5 == 0 or change < 1e-3:
                log(f"    simp q {q:g} it {it:2d} J {J:.6g} grey "
                    f"{np.mean((x > 0.05) & (x < 0.95)):.3f} change {change:.3f} "
                    f"{time.time() - t0:.0f}s")
            if change < 1e-3:
                break
    rel.q = 1.0
    return x


def one(bid: str, power: float, device: str, iters: int) -> dict:
    [b] = common.build_blocks([bid])
    p = lifted.Params(3.0, 8, solver=lifted.solver_of(device))
    c = Clearing(b, 0.5, p, population=common.POPULATIONS["area"])
    rel = Relaxation(c, power)
    t0 = time.time()
    fw = frank_wolfe(rel, D_LENS, iters, log=lambda s: print(f"{bid} {s}", flush=True))
    r = round_by_x(fw["x"], c.cost, D_LENS)
    Jr = rel.exact(r)
    x = fw["x"]
    xs = simp(rel, x, D_LENS, iters // 2, log=lambda s: print(f"{bid} {s}", flush=True))
    rs = round_by_x(xs, c.cost, D_LENS)
    Js = rel.exact(rs)
    out = dict(block=bid, n=c.n, power=power, iters=len(fw["hist"]),
               relaxed_perm=rel.perm(fw["J"]), bound_perm=rel.perm(fw["lb"]),
               rounded_perm=rel.perm(Jr), rounded_D=float(c.cost @ r),
               frac_share=float(np.mean((x > 1e-6) & (x < 1 - 1e-6))),
               frac_cost=float(c.cost[(x > 1e-6) & (x < 1 - 1e-6)].sum()),
               simp_perm=rel.perm(Js), simp_D=float(c.cost @ rs),
               simp_grey=float(np.mean((xs > 0.05) & (xs < 0.95))),
               cleared=np.flatnonzero(r).tolist(), x=x.tolist(),
               simp_cleared=np.flatnonzero(rs).tolist(), simp_x=xs.tolist(),
               t=time.time() - t0)
    print(f"{bid} n={c.n} p={power:g}: relaxed {out['relaxed_perm']:.4f} bound "
          f"{out['bound_perm']:.4f} rounded {out['rounded_perm']:.4f} (D {out['rounded_D']:.4f}) "
          f"fractional cost {out['frac_cost']:.4f}; SIMP rounded {out['simp_perm']:.4f} "
          f"(D {out['simp_D']:.4f}, grey {out['simp_grey']:.3f}); {out['t']:.0f}s", flush=True)
    return out


_CFG: dict = {}


def _run(bid: str) -> None:
    power, device, iters = _CFG["power"], _CFG["device"], _CFG["iters"]
    try:
        pd.DataFrame([one(bid, power, device, iters)]).to_parquet(
            OUT / f"{bid}_p{power:g}.parquet")
    except Exception as e:
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:300], flush=True)


def main(ids: list[str], power: float, device: str, iters: int, workers: int) -> None:
    OUT.mkdir(exist_ok=True)
    _CFG.update(power=power, device=device, iters=iters)
    todo = [i for i in ids if not (OUT / f"{i}_p{power:g}.parquet").exists()]
    if workers == 1:
        for bid in todo:
            _run(bid)
        return
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for _ in pool.imap_unordered(_run, todo):
            pass


if __name__ == "__main__":
    ids = common.recipients() if sys.argv[1] == "all" else sys.argv[1].split(",")
    main(ids, float(sys.argv[2]), sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 40,
         int(sys.argv[5]) if len(sys.argv) > 5 else 1)
