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

    CUDA_PATH=/usr PYTHONPATH=. pixi run python research/roadless/relax.py one|path <ids,|all> <p> <cpu|gpu> <plan> <along> [workers]

one: SIMP at D 0.10 (plus the Frank-Wolfe relaxation and bound if fw > 0), rows in
relax_rows/<along>/<plan>/. path: nested budgets D 0.01 .. 0.15 (each warm-starts from the last
and keeps what it cleared), rows in the greedy's format (clear.rows_dir, picker SIMP<plan>), so
picker_compare / Lens A and B read them like any picker.

plan: fw<FW iterations, 0 = start SIMP from uniform x>.q<final q>.i<OC updates per q>.t<solve
tolerance while optimizing>, e.g. fw40.q3.i20.t1e-05 (the first 220 runs) or fw0.q5.i10.t0.001.
"""
from __future__ import annotations

import dataclasses
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import EPS, RTOL_SCORE, Clearing, _cells_axes, rows_dir  # noqa: E402

D_LENS = 0.10
OUT = HERE / "relax_rows"


class Relaxation:
    """One block's relaxation: open fractions, objective and gradient in x. Building j adds
    x_j^q of its cells' open fraction: q 1 is the convex relaxation (convex for P under the
    uniform conductance), q > 1 (SIMP) makes partial clearing uneconomic (its conductance falls
    faster than its cost). Under a road-favouring along-conductance the factors respond to
    opening too (their vjp), and nothing is convex."""

    def __init__(self, c: Clearing, power: float, q: float = 1.0, rtol: float = RTOL_SCORE,
                 eps: float = EPS):
        self.c, self.power, self.q, self.rtol, self.eps = c, power, q, rtol, eps
        self.beta = 0.0                        # projection sharpness (0: none); simp sets it
        self._u = self._lam = None             # the last solves: warm starts for the next
        g = c.sc.grid
        self.ff0 = g.ff0.ravel()
        self.inside = np.flatnonzero(g.inside)
        self.J0 = c.sc.J(c.sc.u0, power)
        xp = c.p.solver.xp                     # what home_u needs, on the device
        on = np.flatnonzero(c.sc.owner.ravel() >= 0)
        self._on = xp.asarray(on)
        self._owner_on = xp.asarray(c.sc.owner.ravel()[on])
        self._f_on = xp.asarray(c.sc.f.ravel()[on])
        self._share = xp.asarray(lifted.axes(c.p.K)[2] / np.pi)

    def proj(self, x: np.ndarray) -> np.ndarray:
        """The clearing a design x stands for: x itself, or with beta > 0 a smooth step at
        ETA, H(x) = (tanh(beta ETA) + tanh(beta (x - ETA))) / (tanh(beta ETA) + tanh(beta
        (1 - ETA))), so a mostly-closed building conducts ~nothing (no grey leaks) and 0, 1
        stay 0, 1. Conductance share H(x)^q, cost c . H(x)."""
        if self.beta == 0.0:
            return x
        b = self.beta
        return ((np.tanh(b * ETA) + np.tanh(b * (x - ETA)))
                / (np.tanh(b * ETA) + np.tanh(b * (1 - ETA))))

    def dproj(self, x: np.ndarray) -> np.ndarray:
        if self.beta == 0.0:
            return np.ones_like(x)
        b = self.beta
        return b * (1 - np.tanh(b * (x - ETA)) ** 2) / (np.tanh(b * ETA) + np.tanh(b * (1 - ETA)))

    def open_of(self, x: np.ndarray) -> np.ndarray:
        o = self.ff0.copy()
        o[self.inside] += self.c.bf.T @ (self.proj(x) ** self.q)
        return np.minimum(o, self.c.full.ravel()).reshape(self.c.full.shape)

    def _solve(self, x: np.ndarray):
        c = self.c
        op = self.open_of(x)
        op_eps = op + self.eps * (c.full - op)
        sy = lifted.System(c.sc.grid, op_eps, c.p)
        sol = lifted.solve(c.sc.grid, op_eps, c.sc.f, c.p, system=sy, rtol=self.rtol,
                           host=False, x0=self._u)
        self._u = sol.u                        # eps world: the same unknowns every time
        return op_eps, sy, sol, self._home_u(sol, op_eps)

    def _home_u(self, sol: lifted.Solution, op_eps: np.ndarray) -> np.ndarray:
        """Scorer.home_u_of with the per-unknown potentials left on the device (48M of them on
        a big block): each cell's potential by angular share, injection-weighted per home."""
        c, xp = self.c, self.c.p.solver.xp
        free = xp.flatnonzero(xp.asarray(op_eps).ravel() > 0)
        uk = xp.zeros((len(free), c.p.K))
        uk[sol.unk_cell >= 0] = sol.u.reshape(-1, c.p.K)
        ub = xp.zeros(op_eps.size)
        ub[free] = uk @ self._share
        num = c.p.solver.to_host(xp.bincount(self._owner_on, weights=self._f_on * ub[self._on],
                                             minlength=c.n))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c.sc.live_home, num / c.sc.w, np.nan)

    def value(self, x: np.ndarray) -> float:
        return self.c.sc.J(self._solve(x)[3], self.power)

    def value_sgrad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """J and its gradient in the projected clearing s = proj(x)."""
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
            lam = lifted.solve(g, op_eps, c.sc.f * mult, p, system=sy, rtol=self.rtol,
                               host=False, x0=self._lam).u
            self._lam = lam
        K = p.K
        uk, lk = _cells_axes(sy, sol.u, K), _cells_axes(sy, lam, K)
        o = xp.asarray(op_eps).ravel()
        cell, _nf = lifted._free_ids(o, xp)
        pat = lifted.pattern(g, K, xp)
        v, _th, m, gap = lifted.axes(K)
        gc = xp.zeros(o.size)                           # dJ / d(open) per grid cell
        Fl = p.along.layers(op_eps, g.h, K)
        dF = None                                       # dJ / d(layer factor), if any vary
        for k in range(K):
            a, b, line = pat.along[k]
            cells = [a, b, *list(line)]
            vals = xp.stack([o[q] for q in cells])
            frac = vals.min(axis=0)
            act = vals <= frac[None, :]                  # the cells attaining the min
            ca, cb = cell[a], cell[b]
            q = (uk[ca, k] - uk[cb, k]) * (lk[ca, k] - lk[cb, k])
            dx, dy = int(v[k, 0]), int(v[k, 1])
            const = m[k] / float(dx * dx + dy * dy)
            if np.isscalar(Fl[k]):
                boost = Fl[k]
            else:                                       # w = frac const min(F_a, F_b)
                Fk = xp.asarray(Fl[k]).ravel()
                fa, fb = Fk[a], Fk[b]
                boost = xp.minimum(fa, fb)
                if dF is None:
                    dF = xp.zeros((K, o.size))
                dF[k] += xp.bincount(xp.where(fa <= fb, a, b), -q * frac * const,
                                     minlength=o.size)
            coef = -q * const * boost / act.sum(axis=0)
            for i, qc in enumerate(cells):
                gc += xp.bincount(qc[act[i]], coef[act[i]], minlength=o.size)
        if dF is not None:                              # the factors' own response to opening
            gc += xp.asarray(p.along.vjp(op_eps, g.h, K, dF.reshape(K, *op_eps.shape))).ravel()
        free = xp.flatnonzero(o > 0)
        for k in range(K):
            k2 = (k + 1) % K
            q = (uk[:, k] - uk[:, k2]) * (lk[:, k] - lk[:, k2])
            gc[free] += -q * (g.h * g.h / (p.ell_m ** 2 * gap[k]))
        grad = (1.0 - self.eps) * (c.bf @ p.solver.to_host(gc)[self.inside])
        if self.q != 1.0:
            grad = grad * self.q * self.proj(x) ** (self.q - 1.0)
        return J, grad

    def value_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        J, gs = self.value_sgrad(x)
        return J, gs * self.dproj(x)

    def exact(self, x: np.ndarray) -> float:
        """J_p of a 0/1 clearing in the real geometry (buildings closed)."""
        return self.exact_JP(x)[0]

    def exact_JP(self, x: np.ndarray) -> tuple[float, float]:
        """(J_p, P) of a 0/1 clearing in the real geometry (buildings closed)."""
        op = self.open_of(x)
        sol = lifted.solve(self.c.sc.grid, op, self.c.sc.f, self.c.p, rtol=RTOL_SCORE)
        return self.c.sc.J(self.c.sc.home_u_of(sol, op), self.power), sol.P

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


def round_by_x(x: np.ndarray, cost: np.ndarray, budget: float,
               first: np.ndarray | None = None) -> np.ndarray:
    """Clear buildings by decreasing x while they fit the budget (`first` before any other)."""
    r = np.zeros(len(x))
    left = budget + 1e-12
    key = -x if first is None else -(x + 2.0 * first)
    for j in np.argsort(key, kind="stable"):
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
ETA = 0.5                                          # the projection's threshold


class Plan(NamedTuple):
    """How a block is solved: `fw` Frank-Wolfe iterations from x = 0 (0: start SIMP from the
    uniform x = D, no bound), SIMP continuation q = 1.5, 2, ... `qmax` with up to `iters`
    optimality-criteria updates each, solves to `rtol` while optimizing (scores are exact),
    closed buildings conducting `eps` while optimizing (named only when not the tension's),
    and projected (Relaxation.proj) with beta from `b0` doubling each stage up to `bmax`,
    extra stages at qmax until it gets there (b0 0: no projection, not named); `keep` > 0:
    the answer is the best exact-scored rounding of the iterates every `keep` updates (0: the
    last iterate's); `kscore` > 0 scores the candidates in the eps world at that eps (the
    relaxation's cached structure: no new system per candidate), the winner exactly; `warm`
    > 0 (path only): each budget after the first starts from the last budget's x (the free
    buildings lifted to the uniform share) with `warm` updates per stage."""
    fw: int
    qmax: float
    iters: int
    rtol: float
    eps: float = EPS
    b0: float = 0.0
    bmax: float = 0.0
    keep: int = 0
    kscore: float = 0.0
    warm: int = 0

    @property
    def name(self) -> str:
        return (f"fw{self.fw}.q{self.qmax:g}.i{self.iters}.t{self.rtol:g}"
                + (f".e{self.eps:g}" if self.eps != EPS else "")
                + (f".b{self.b0:g}-{self.bmax:g}" if self.b0 > 0 else "")
                + (f".k{self.keep}" if self.keep > 0 else "")
                + (f"s{self.kscore:g}" if self.kscore > 0 else "")
                + (f".w{self.warm}" if self.warm > 0 else ""))

    @property
    def stages(self) -> tuple[tuple[float, float], ...]:
        """(q, beta) per continuation stage."""
        qs = [float(q) for q in np.arange(1.5, self.qmax + 1e-9, 0.5)]
        if self.b0 == 0:
            return tuple((q, 0.0) for q in qs)
        out, b = [], self.b0
        for q in qs:
            out.append((q, b))
            b = min(2 * b, self.bmax)
        while out[-1][1] < self.bmax:
            out.append((qs[-1], b))
            b = min(2 * b, self.bmax)
        return tuple(out)


def plan_of(spec: str) -> Plan:
    m = re.fullmatch(r"fw(\d+)\.q([0-9.]+)\.i(\d+)\.t([0-9.e-]+?)(?:\.e([0-9.e-]+?))?"
                     r"(?:\.b([0-9.]+)-([0-9.]+))?(?:\.k(\d+)(?:s([0-9.e-]+))?)?(?:\.w(\d+))?", spec)
    if m is None:
        raise ValueError(f"unknown plan {spec!r}")
    return Plan(int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4)),
                EPS if m.group(5) is None else float(m.group(5)),
                0.0 if m.group(6) is None else float(m.group(6)),
                0.0 if m.group(7) is None else float(m.group(7)),
                0 if m.group(8) is None else int(m.group(8)),
                0.0 if m.group(9) is None else float(m.group(9)),
                0 if m.group(10) is None else int(m.group(10)))


class Incumbent:
    """The best 0/1 clearing among SIMP's iterates, each `every`-th rounded (`first` cleared
    before any other) and scored exactly. SIMP's iterates can hold a gate and lose it: on 22422
    the rounding of update 3 scores 0.810 and the last 0.012, the gate and its neighbours, being
    substitutes, flipping between x 0.3 and 0.5 until the continuation drops them (NOTES)."""

    def __init__(self, rel: Relaxation, budget: float, every: int,
                 first: np.ndarray | None = None, score_eps: float = 0.0):
        self.rel, self.budget, self.every, self.first = rel, budget, every, first
        self.score_eps = score_eps
        self.n = 0
        self.J, self.r = np.inf, np.zeros(rel.c.n)
        self._seen: set[bytes] = set()

    def _score(self, r: np.ndarray) -> float:
        if self.score_eps == 0.0:
            return self.rel.exact(r)
        rel = self.rel                  # the eps world at score_eps, q 1, no projection
        saved = rel.eps, rel.q, rel.beta
        rel.eps, rel.q, rel.beta = self.score_eps, 1.0, 0.0
        try:
            return rel.value(r)
        finally:
            rel.eps, rel.q, rel.beta = saved

    def offer(self, x: np.ndarray) -> None:
        r = round_by_x(x, self.rel.c.cost, self.budget, first=self.first)
        key = np.packbits(r > 0).tobytes()
        if key in self._seen:
            return
        self._seen.add(key)
        J = self._score(r)
        if J < self.J:
            self.J, self.r = J, r

    def best(self) -> tuple[float, np.ndarray]:
        """(exact J, clearing) of the best candidate."""
        return (self.J if self.score_eps == 0.0 else self.rel.exact(self.r)), self.r

    def __call__(self, x: np.ndarray) -> None:
        self.n += 1
        if self.n % self.every == 0:
            self.offer(x)


def simp(rel: Relaxation, x0: np.ndarray, budget: float,
         stages: tuple[tuple[float, float], ...], iters: int, log=print, move: float = 0.2,
         xlo: np.ndarray | None = None, watch=None) -> np.ndarray:
    """SIMP continuation from x0: for (q, beta) in stages, up to `iters` optimality-criteria
    updates x <- clip(x (-g / (lam c H'(x)))^1/2) within `move` of x and [xlo (default XMIN),
    1], lam bisected so that c . H(x) = budget (H = rel.proj; H' cancels the one in g, so the
    ratio is the sensitivity to the projected clearing per unit cost). Non-convex: a local
    method. `watch(x)`, if given, sees every iterate."""
    cost = rel.c.cost
    lo_ = np.full(len(cost), XMIN) if xlo is None else xlo
    x = np.clip(x0, lo_, 1.0)
    free = x > lo_
    rel.q, rel.beta = stages[0]
    over = float(cost @ rel.proj(x)) - budget
    if over > 0:                                         # scale the free part into the budget
        x[free] = np.maximum(lo_[free], x[free] * (1 - over / float(cost[free] @ x[free])))
    t0 = time.time()
    for q, beta in stages:
        rel.q, rel.beta = q, beta
        for it in range(iters):
            J, gs = rel.value_sgrad(x)
            B = np.maximum(-gs, 1e-300) / cost
            lo, hi = 1e-300, 1e300
            for _ in range(200):
                lam = np.sqrt(lo * hi)
                xn = np.clip(x * np.sqrt(B / lam), np.maximum(lo_, x - move),
                             np.minimum(1.0, x + move))
                lo, hi = (lam, hi) if cost @ rel.proj(xn) > budget else (lo, lam)
                if hi / lo < 1 + 1e-9:
                    break
            change = float(np.abs(xn - x).max())
            x = xn
            if watch is not None:
                watch(x)
            if it % 5 == 0 or change < 1e-3:
                sx = rel.proj(x)
                log(f"    simp q {q:g} beta {beta:g} it {it:2d} J {J:.6g} grey "
                    f"{np.mean((sx > 0.05) & (sx < 0.95)):.3f} change {change:.3f} "
                    f"{time.time() - t0:.0f}s")
            if change < 1e-3:
                break
    rel.q, rel.beta = 1.0, 0.0
    return x


def _clearing(bid: str, device: str, along: str) -> Clearing:
    [b] = common.build_blocks([bid])
    p = lifted.Params(3.0, 8, along=lifted.along_of(along, lifted.scans_of(device)),
                      solver=lifted.solver_of(device))
    return Clearing(b, 0.5, p, population=common.POPULATIONS["area"])


def one(bid: str, power: float, device: str, plan: Plan, along: str) -> dict:
    c = _clearing(bid, device, along)
    rel = Relaxation(c, power, rtol=plan.rtol, eps=plan.eps)
    log = lambda s: print(f"{bid} {s}", flush=True)  # noqa: E731
    t0 = time.time()
    out = dict(block=bid, n=c.n, power=power, plan=plan.name, along=along)
    if plan.fw > 0:
        fw = frank_wolfe(rel, D_LENS, plan.fw, log=log)
        x = fw["x"]
        r = round_by_x(x, c.cost, D_LENS)
        frac = (x > 1e-6) & (x < 1 - 1e-6)
        out.update(iters=len(fw["hist"]), relaxed_perm=rel.perm(fw["J"]),
                   bound_perm=rel.perm(fw["lb"]), rounded_perm=rel.perm(rel.exact(r)),
                   rounded_D=float(c.cost @ r), frac_share=float(np.mean(frac)),
                   frac_cost=float(c.cost[frac].sum()), cleared=np.flatnonzero(r).tolist(),
                   x=x.tolist())
    else:
        x = np.full(c.n, D_LENS)
    inc = Incumbent(rel, D_LENS, max(plan.keep, 1), score_eps=plan.kscore)
    xs = simp(rel, x, D_LENS, plan.stages, plan.iters, log=log,
              watch=inc if plan.keep > 0 else None)
    inc.offer(xs)                                         # the last iterate's rounding
    Jbest, rs = inc.best()
    out.update(simp_perm=rel.perm(Jbest), simp_D=float(c.cost @ rs),
               simp_grey=float(np.mean((xs > 0.05) & (xs < 0.95))),
               simp_cleared=np.flatnonzero(rs).tolist(), simp_x=xs.tolist(),
               t=time.time() - t0)
    print(f"{bid} n={c.n} p={power:g} {along} {plan.name}: "
          + (f"relaxed {out['relaxed_perm']:.4f} bound {out['bound_perm']:.4f} rounded "
             f"{out['rounded_perm']:.4f}; " if plan.fw > 0 else "")
          + f"SIMP rounded {out['simp_perm']:.4f} (D {out['simp_D']:.4f}, grey "
          f"{out['simp_grey']:.3f}); {out['t']:.0f}s", flush=True)
    return out


PATH_DS = tuple(round(0.01 * k, 2) for k in range(1, 16))


def path(bid: str, power: float, device: str, plan: Plan, along: str) -> pd.DataFrame:
    """Nested SIMP: for D in PATH_DS, everything cleared so far held at 1, the rest reset to an
    even share of the remaining budget and the full q continuation rerun (warm-starting the
    rest at q = qmax traps them: a building SIMP has switched off, x = XMIN, has gradient
    ~ q x^(q-1) ~ 0 and multiplicative updates, so it never comes back -- measured: -0.076 Lens
    A against the greedy), round, score exactly. plan.warm > 0: after the first budget, the
    rest start from the last x lifted to at least that even share (so none is trapped at
    XMIN), and each stage gets plan.warm updates. Rows in the greedy's format: `cleared` = the
    buildings new at this step."""
    c = _clearing(bid, device, along)
    rel = Relaxation(c, power, rtol=plan.rtol, eps=plan.eps)
    J0, P0 = rel.J0, c.sc.P0
    rows = [dict(block=bid, n=c.n, step=0, D=0.0, perm=0.0, perm1=0.0, cleared=[], P0=P0,
                 t=0.0)]
    fixed = np.zeros(c.n, dtype=bool)
    x = np.full(c.n, PATH_DS[0])
    t0 = time.time()
    for i, D in enumerate(PATH_DS):
        rest = max(D - float(c.cost[fixed].sum()), 0.0)
        even = rest / max(float(c.cost[~fixed].sum()), 1e-12)
        warm = plan.warm > 0 and i > 0
        x = np.where(fixed, 1.0, np.maximum(x, even) if warm else even)
        inc = Incumbent(rel, D, max(plan.keep, 1), first=fixed.astype(float),
                        score_eps=plan.kscore)
        x = simp(rel, x, D, plan.stages, plan.warm if warm else plan.iters, log=lambda s: None,
                 xlo=np.where(fixed, 1.0, XMIN), watch=inc if plan.keep > 0 else None)
        inc.offer(x)
        r = inc.r > 0
        assert r[fixed].all()
        new = np.flatnonzero(r & ~fixed)
        fixed |= r
        J, P = rel.exact_JP(fixed.astype(float))
        rows.append(dict(block=bid, n=c.n, step=len(rows), D=float(c.cost[fixed].sum()),
                         perm=1 - (J / J0) ** (1 / power), perm1=1 - P / P0,
                         cleared=new.tolist(), P0=P0, t=time.time() - t0))
        print(f"  {bid} D {D:.2f}: {rows[-1]['D']:.4f} perm' {rows[-1]['perm']:.4f} "
              f"{rows[-1]['t']:.0f}s", flush=True)
    return pd.DataFrame(rows)


def rows_of(plan: Plan, along: str) -> Path:
    return OUT / along / plan.name


_CFG: dict = {}


def _run(bid: str) -> None:
    mode, power, device, plan, along = (_CFG[k] for k in ("mode", "power", "device", "plan",
                                                          "along"))
    try:
        if mode == "one":
            pd.DataFrame([one(bid, power, device, plan, along)]).to_parquet(
                rows_of(plan, along) / f"{bid}_p{power:g}.parquet")
        else:
            out = rows_dir(f"SIMP{plan.name}", 0.5, "area", power, along) / f"{bid}.parquet"
            path(bid, power, device, plan, along).to_parquet(out)
            print(f"{time.strftime('%H:%M:%S')} {bid} path done", flush=True)
    except Exception as e:
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:300], flush=True)


def _target(mode: str, bid: str, power: float, plan: Plan, along: str) -> Path:
    if mode == "one":
        return rows_of(plan, along) / f"{bid}_p{power:g}.parquet"
    return rows_dir(f"SIMP{plan.name}", 0.5, "area", power, along) / f"{bid}.parquet"


def main(mode: str, ids: list[str], power: float, device: str, plan: Plan, along: str,
         workers: int) -> None:
    if mode not in ("one", "path"):
        raise ValueError(f"unknown mode {mode!r}")
    _target(mode, ids[0], power, plan, along).parent.mkdir(parents=True, exist_ok=True)
    _CFG.update(mode=mode, power=power, device=device, plan=plan, along=along)
    todo = [i for i in ids if not _target(mode, i, power, plan, along).exists()]
    if workers == 1:
        for bid in todo:
            _run(bid)
        return
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for _ in pool.imap_unordered(_run, todo):
            pass


if __name__ == "__main__":
    ids = common.recipients() if sys.argv[2] == "all" else sys.argv[2].split(",")
    main(sys.argv[1], ids, float(sys.argv[3]), sys.argv[4], plan_of(sys.argv[5]), sys.argv[6],
         int(sys.argv[7]) if len(sys.argv) > 7 else 1)
