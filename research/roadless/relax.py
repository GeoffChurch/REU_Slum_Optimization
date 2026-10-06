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

    CUDA_PATH=/usr PYTHONPATH=. uv run python research/roadless/relax.py one|path <ids,|all> <p> <cpu|gpu> <plan> <along> [workers] [budget]

one: SIMP at one budget (default D_LENS, 0.05; plus the Frank-Wolfe relaxation and bound if
fw > 0), rows in relax_rows/<along>/<plan>/D<budget>/. path: nested budgets D 0.01 .. 0.15,
rows in the greedy's format (clear.rows_dir, picker SIMP<plan>), so picker_compare reads them
like any picker.

plan: fw<FW iterations, 0 = start SIMP from uniform x>.q<final q>.i<OC updates per q>.t<solve
tolerance while optimizing>, e.g. fw40.q3.i20.t1e-05 (the first 220 runs) or fw0.q5.i10.t0.001;
the optional suffixes are Plan's (.e .b .k[s] .r[l] .p .w .c .u).
"""
from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
import clear  # noqa: E402
from clear import EPS, RTOL_SCORE, Clearing, _cells_axes, rows_dir  # noqa: E402

D_LENS = 0.05          # Lens A's budget (owner, 2026-10-03: 0.10 saturates under sightline)
OUT = HERE / "relax_rows"


class Relaxation:
    """One block's relaxation: open fractions, objective and gradient in x. Building j adds
    x_j^q of its cells' open fraction: q 1 is the convex relaxation (convex for P under the
    uniform conductance), q > 1 (SIMP) makes partial clearing uneconomic (its conductance falls
    faster than its cost). Under a road-favouring along-conductance the factors respond to
    opening too (their vjp), and nothing is convex."""

    def __init__(self, c: Clearing, power: float, q: float = 1.0, rtol: float = RTOL_SCORE,
                 eps: float = EPS, params: lifted.Params | None = None):
        self.c, self.power, self.q, self.rtol, self.eps = c, power, q, rtol, eps
        # the conductance optimized under: the clearing's search one (c.ps, e.g. more
        # translucent than the metric), or `params`; exact() always scores under the metric c.p
        self.pp = c.ps if params is None else params
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
        sy = lifted.System(c.sc.grid, op_eps, self.pp)
        sol = lifted.solve(c.sc.grid, op_eps, c.sc.f, self.pp, system=sy, rtol=self.rtol,
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
        c, p = self.c, self.pp
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


def cut_to_budget(order: list[int], cost: np.ndarray, budget: float) -> np.ndarray:
    """A greedy's clearing within the budget: every building of its pick order while it fits."""
    r = np.zeros(len(cost))
    left = budget + 1e-12
    for j in order:
        if cost[j] <= left:
            r[j] = 1.0
            left -= cost[j]
    return r


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


# The randomized roundings' generator seed: fixed, part of the method a plan's `.r<n>` names.
SAMPLE_SEED = 0
# Multi-start's: start s > 0 draws from START_SEED + s (`.m<k>`).
START_SEED = 1000


def start_x(s: int, cost: np.ndarray, budget: float) -> np.ndarray:
    """SIMP's start s: s 0 the uniform x = D (cost . x = D, the costs summing to 1); s > 0
    exponential weights scaled onto the budget, seeded (START_SEED + s)."""
    if s == 0:
        return np.full(len(cost), budget)
    w = np.random.default_rng(START_SEED + s).exponential(size=len(cost))
    return budget * w / float(cost @ w)


def round_sampled(x: np.ndarray, cost: np.ndarray, budget: float, rng: np.random.Generator,
                  first: np.ndarray | None = None) -> np.ndarray:
    """A randomized rounding: SIMP's live buildings (x above twice its floor XMIN) in a random
    order drawn with probability proportional to x (Plackett-Luce, by Gumbel keys), then the
    buildings at the floor by decreasing x as round_by_x takes them, cleared while they fit the
    budget (`first` before any other). The floor sits out of the draw: hundreds of buildings at
    XMIN outweigh a few ones and would push them out of the budget. So a near-binary x keeps its
    ones, and grey buildings -- a gate and its substitutes flipping between x 0.3 and 0.5 --
    come in different orders."""
    live = x > 2 * XMIN
    tier = np.where(live, 1, 2) if first is None else np.where(first > 0, 0, np.where(live, 1, 2))
    gumbel = np.zeros(len(x))
    gumbel[live] = -(np.log(x[live]) + rng.gumbel(size=int(live.sum())))
    order = np.lexsort((np.where(live, gumbel, -x), tier))     # by tier, then key
    r = np.zeros(len(x))
    left = budget + 1e-12
    for j in order:
        if x[j] <= 1e-9:
            break
        if cost[j] <= left:
            r[j] = 1.0
            left -= cost[j]
    return r


# Pair moves draw from the PAIR_POOL most promising buildings to add and to close.
PAIR_POOL = 64


def _singles(g, ins, outs, cost, left):
    """Add one (budget left permitting) or swap one in for one out: moves as rows (added,
    added, closed, closed; -1 for none) and their linearized changes."""
    ia, ib = np.meshgrid(ins, outs, indexing="ij")
    none_s, none_p = np.full(len(ins), -1), np.full(ia.size, -1)
    mv = np.concatenate([np.stack([ins, none_s, none_s, none_s]),
                         np.stack([ia.ravel(), none_p, ib.ravel(), none_p])], axis=1)
    est = np.concatenate([np.where(cost[ins] <= left, g[ins], np.inf),
                          np.where(cost[ia] <= left + cost[ib], g[ia] - g[ib], np.inf).ravel()])
    return mv, est


def _pairs(g, ins, outs, cost, left):
    """Two in for one out and one in for two out, among the PAIR_POOL buildings most promising
    to add (most negative gradient) and cheapest to close."""
    pin = ins[np.argsort(g[ins], kind="stable")[:PAIR_POOL]]
    pout = outs[np.argsort(-g[outs], kind="stable")[:PAIR_POOL]]
    i, j = np.triu_indices(len(pin), 1)
    a1, a2, b = np.repeat(pin[i], len(pout)), np.repeat(pin[j], len(pout)), np.tile(pout, len(i))
    k, m = np.triu_indices(len(pout), 1)
    a, b1, b2 = np.repeat(pin, len(k)), np.tile(pout[k], len(pin)), np.tile(pout[m], len(pin))
    mv = np.concatenate([np.stack([a1, a2, b, np.full(len(b), -1)]),
                         np.stack([a, np.full(len(a), -1), b1, b2])], axis=1)
    est = np.concatenate([
        np.where(cost[a1] + cost[a2] <= left + cost[b], g[a1] + g[a2] - g[b], np.inf),
        np.where(cost[a] <= left + cost[b1] + cost[b2], g[a] - g[b1] - g[b2], np.inf)])
    return mv, est


def exchange(score: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray],
             J: float, r: np.ndarray, movable: np.ndarray, cost: np.ndarray, budget: float,
             tries: int, width: int, pairs: bool) -> tuple[float, np.ndarray, int, int]:
    """Exchange refinement of the 0/1 clearing r (score J) over the buildings `movable`. Each
    round ranks the single moves within them -- add one the budget left allows, or swap one in
    for a cleared one -- by their linearized change from grad(r) (adding a: g_a; closing b:
    -g_b), scores the best `width` and keeps the best of those if it improves. With `pairs`, a
    round whose singles improve nothing then tries the best `width` two-in-one-out and
    one-in-two-out moves (_pairs; floating search's escalation: their summed estimates would
    crowd the singles out of one ranking). Stops when a round improves nothing or `tries`
    scorings are spent. Returns (J, r, moves kept, scorings)."""
    used = moves = 0
    while used < tries:
        g = grad(r)
        left = budget + 1e-12 - float(cost @ r)
        ins, outs = movable[r[movable] == 0], movable[r[movable] > 0]
        best = None
        for kind in (_singles, _pairs) if pairs else (_singles,):
            mv, e = kind(g, ins, outs, cost, left)
            k = min(width, tries - used, int(np.isfinite(e).sum()))
            if k == 0:
                continue
            top = np.argpartition(e, k - 1)[:k]
            for m in top[np.argsort(e[top])]:
                rn = r.copy()
                for x in mv[:2, m]:
                    if x >= 0:
                        rn[x] = 1.0
                for x in mv[2:, m]:
                    if x >= 0:
                        rn[x] = 0.0
                used += 1
                Jn = score(rn)
                if Jn < J and (best is None or Jn < best[0]):
                    best = (Jn, rn)
            if best is not None:
                break
        if best is None:
            break
        J, r = best
        moves += 1
    return J, r, moves, used


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


class Stepper(Protocol):
    def trial(self, x: np.ndarray, B: np.ndarray, lo: np.ndarray
              ) -> Callable[[float], np.ndarray]:
        """The update of iterate x, as a function of the budget's multiplier lam (simp bisects
        lam so the update spends the budget). B = -dJ/dx / cost, the sensitivity per unit
        cost; lo the lower bounds."""
        ...


class Update(Protocol):
    """SIMP's design update rule (the plan's `.u<name>`): `start()` gives a fresh stepper per
    continuation stage (its objective changes with q)."""
    @property
    def name(self) -> str: ...

    def start(self) -> Stepper: ...


@dataclass(frozen=True)
class OC:
    """Optimality criteria: x (B / lam)^eta within `move` of x. eta 0.5 is the classic rule
    (the stationarity condition of the linearized Lagrangian in x^2); a smaller eta damps it."""
    eta: float = 0.5
    move: float = 0.2

    @property
    def name(self) -> str:
        return f"oc{self.eta:g}m{self.move:g}"

    def start(self) -> Stepper:
        return self

    def trial(self, x, B, lo):
        a, b = np.maximum(lo, x - self.move), np.minimum(1.0, x + self.move)
        return lambda lam: np.clip(x * (B / lam) ** self.eta, a, b)


@dataclass(frozen=True)
class MMA:
    """Method of moving asymptotes (Svanberg 1987) for this problem: J is non-increasing in x,
    so its convex approximation needs only the lower asymptote L, and with the budget exact the
    update is L + (x - L) sqrt(B / lam) -- OC is this with L fixed at 0. Each building's gap
    x - L starts at x (the first step is OC's), then shrinks by `shrink` when its last two moves
    reversed (an oscillation: a gate and its substitutes flipping) and grows by `grow` when they
    agree (a steady climb, additive once L < 0, so a building near XMIN can come back). Moves
    stay within 0.9 gap and `move` of x."""
    move: float = 0.2
    shrink: float = 0.7
    grow: float = 1.2

    @property
    def name(self) -> str:
        return f"mma{self.move:g}"

    def start(self) -> Stepper:
        return _MMAStep(self)


class _MMAStep:
    def __init__(self, rule: MMA):
        self.rule = rule
        self.prev: list[np.ndarray] = []        # the last two iterates
        self.gap: np.ndarray | None = None

    def trial(self, x, B, lo):
        r = self.rule
        if self.gap is None or len(self.prev) < 2:
            gap = np.maximum(x, 0.01) if self.gap is None else self.gap
        else:
            turn = (x - self.prev[-1]) * (self.prev[-1] - self.prev[-2])
            gap = self.gap * np.where(turn < 0, r.shrink, np.where(turn > 0, r.grow, 1.0))
        self.gap = gap = np.clip(gap, 0.01, 10.0)
        self.prev = [*self.prev[-1:], x.copy()]
        L = x - gap
        a = np.maximum(lo, np.maximum(x - r.move, x - 0.9 * gap))
        b = np.minimum(1.0, np.minimum(x + r.move, x + 0.9 * gap))
        return lambda lam: np.clip(L + gap * np.sqrt(B / lam), a, b)


DEFAULT_UPDATE = OC()


class Plan(NamedTuple):
    """How a block is solved: `fw` Frank-Wolfe iterations from x = 0 (0: start SIMP from the
    uniform x = D, no bound), SIMP continuation q = 1.5, 2, ... `qmax` with up to `iters`
    optimality-criteria updates each, solves to `rtol` while optimizing (scores are exact),
    closed buildings conducting `eps` while optimizing (named only when not the tension's),
    and projected (Relaxation.proj) with beta from `b0` doubling each stage up to `bmax`,
    extra stages at qmax until it gets there (b0 0: no projection, not named); `keep` > 0:
    the answer is the best exact-scored rounding of the iterates every `keep` updates (0: the
    last iterate's); `kscore` > 0 scores the candidates in the eps world at that eps (the
    relaxation's cached structure: no new system per candidate), the winner exactly; `samples`
    > 0: each offered iterate is also rounded that many times at random (round_sampled), every
    distinct candidate scored, in every stage or (`late` > 0) only the last `late`; `polish`
    > 0: then exchange refinement on the undecided buildings (Incumbent.polish), `polish`
    scorings at most, the best `pwidth` ranked moves scored per round, over every building
    in each phase of `polish_sets` (True: every building, `P`; False: the undecided ones,
    `p`; `pP` the undecided then every one), with two-for-one moves if `polish_pairs` (`x2`);
    `seed` (one only): that greedy picker's clearing within the budget offered to the incumbent
    before SIMP runs (`.g<picker>`: the polish starts from the better of the two); `warm`
    > 0 (path only): each budget after the first starts from the last budget's x (the free
    buildings lifted to the uniform share) with `warm` updates per stage; `coarse` > 0 (one
    only): every stage but the last on a grid of that spacing (h 1.0: 4x fewer unknowns), the
    last at h 0.5 from its x (buildings, their costs and x do not depend on h), the incumbent
    scoring at h 0.5 throughout; `coarse_all` > 0 (one only): every stage on that grid, each
    start's incumbent scored there and only its winner at h 0.5; `starts` > 1 (one only): that
    many SIMP runs (start_x), one incumbent over all of them (coarse_all: over their winners);
    `update` the design update rule (`.u<name>`; OC's classic rule
    is not named)."""
    fw: int
    qmax: float
    iters: int
    rtol: float
    eps: float = EPS
    b0: float = 0.0
    bmax: float = 0.0
    keep: int = 0
    kscore: float = 0.0
    samples: int = 0
    late: int = 0
    polish: int = 0
    pwidth: int = 0
    polish_sets: tuple[bool, ...] = ()
    polish_pairs: bool = False
    warm: int = 0
    coarse: float = 0.0
    coarse_all: float = 0.0
    starts: int = 0
    seed: str = ""
    update: Update = DEFAULT_UPDATE

    @property
    def name(self) -> str:
        return (f"fw{self.fw}.q{self.qmax:g}.i{self.iters}.t{self.rtol:g}"
                + (f".e{self.eps:g}" if self.eps != EPS else "")
                + (f".b{self.b0:g}-{self.bmax:g}" if self.b0 > 0 else "")
                + (f".k{self.keep}" if self.keep > 0 else "")
                + (f"s{self.kscore:g}" if self.kscore > 0 else "")
                + (f".r{self.samples}" if self.samples > 0 else "")
                + (f"l{self.late}" if self.late > 0 else "")
                + (f".{''.join('P' if a else 'p' for a in self.polish_sets)}{self.polish}"
                   f"w{self.pwidth}" + ("x2" if self.polish_pairs else "")
                   if self.polish > 0 else "")
                + (f".w{self.warm}" if self.warm > 0 else "")
                + (f".c{self.coarse:g}" if self.coarse > 0 else "")
                + (f".C{self.coarse_all:g}" if self.coarse_all > 0 else "")
                + (f".m{self.starts}" if self.starts > 1 else "")
                + (f".g{self.seed}" if self.seed else "")
                + (f".u{self.update.name}" if self.update != DEFAULT_UPDATE else ""))

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
    m = re.fullmatch(r"fw(?P<fw>\d+)\.q(?P<q>[0-9.]+)\.i(?P<i>\d+)\.t(?P<t>[0-9.e-]+?)"
                     r"(?:\.e(?P<e>[0-9.e-]+?))?(?:\.b(?P<b0>[0-9.]+)-(?P<bmax>[0-9.]+))?"
                     r"(?:\.k(?P<k>\d+)(?:s(?P<ks>[0-9.e-]+))?)?(?:\.r(?P<r>\d+)(?:l(?P<rl>\d+))?)?"
                     r"(?:\.(?P<pset>p|P|pP)(?P<p>\d+)w(?P<pw>\d+)(?P<px>x2)?)?"
                     r"(?:\.w(?P<w>\d+))?(?:\.c(?P<c>[0-9.]+))?(?:\.C(?P<C>[0-9.]+))?(?:\.m(?P<m>\d+))?"
                     r"(?:\.g(?P<g>S[0-9.]+cat(?:w\d+)?))?"
                     r"(?:\.u(?:oc(?P<oc>[0-9.]+)m(?P<ocm>[0-9.]+)|mma(?P<mma>[0-9.]+)))?", spec)
    if m is None:
        raise ValueError(f"unknown plan {spec!r}")
    g = m.groupdict()
    update: Update = (OC(float(g["oc"]), float(g["ocm"])) if g["oc"] is not None
                      else MMA(float(g["mma"])) if g["mma"] is not None
                      else DEFAULT_UPDATE)
    return Plan(fw=int(g["fw"]), qmax=float(g["q"]), iters=int(g["i"]), rtol=float(g["t"]),
                eps=EPS if g["e"] is None else float(g["e"]),
                b0=0.0 if g["b0"] is None else float(g["b0"]),
                bmax=0.0 if g["bmax"] is None else float(g["bmax"]),
                keep=0 if g["k"] is None else int(g["k"]),
                kscore=0.0 if g["ks"] is None else float(g["ks"]),
                samples=0 if g["r"] is None else int(g["r"]),
                late=0 if g["rl"] is None else int(g["rl"]),
                polish=0 if g["p"] is None else int(g["p"]),
                pwidth=0 if g["pw"] is None else int(g["pw"]),
                polish_sets=() if g["pset"] is None else tuple(ch == "P" for ch in g["pset"]),
                polish_pairs=g["px"] is not None,
                warm=0 if g["w"] is None else int(g["w"]),
                coarse=0.0 if g["c"] is None else float(g["c"]),
                coarse_all=0.0 if g["C"] is None else float(g["C"]),
                starts=0 if g["m"] is None else int(g["m"]), seed=g["g"] or "", update=update)


class Incumbent:
    """The best 0/1 clearing among SIMP's iterates, each `every`-th rounded (`first` cleared
    before any other) and scored exactly. SIMP's iterates can hold a gate and lose it: on 22422
    the rounding of update 3 scores 0.810 and the last 0.012, the gate and its neighbours, being
    substitutes, flipping between x 0.3 and 0.5 until the continuation drops them (NOTES)."""

    def __init__(self, rel: Relaxation, budget: float, every: int, *, samples: int,
                 sample_in: tuple[tuple[float, float], ...],
                 first: np.ndarray | None = None, score_eps: float = 0.0):
        self.rel, self.budget, self.every, self.first = rel, budget, every, first
        # samples > 0: each offered iterate of a stage in `sample_in` ((q, beta)) is also
        # rounded that many times at random
        self.samples, self.sample_in = samples, sample_in
        self.rng = np.random.default_rng(SAMPLE_SEED)
        # score_eps > 0: candidates scored in the eps world at score_eps under the METRIC
        # conductance (rel may optimize under a search one), q 1, no projection
        self.scorer = (Relaxation(rel.c, rel.power, q=1.0, rtol=rel.rtol, eps=score_eps,
                                  params=rel.c.p) if score_eps > 0 else None)
        self.n = 0
        self.J, self.r = np.inf, np.zeros(rel.c.n)
        self._seen: set[bytes] = set()

    def _score(self, r: np.ndarray) -> float:
        return self.rel.exact(r) if self.scorer is None else self.scorer.value(r)

    def offer(self, x: np.ndarray, stage: tuple[float, float]) -> None:
        cost = self.rel.c.cost
        grey = float(np.mean((x > 0.05) & (x < 0.95)))
        n = self.samples if stage in self.sample_in else 0
        for k, r in enumerate([round_by_x(x, cost, self.budget, first=self.first),
                               *(round_sampled(x, cost, self.budget, self.rng, first=self.first)
                                 for _ in range(n))]):
            key = np.packbits(r > 0).tobytes()
            if key in self._seen:
                continue
            self._seen.add(key)
            J = self._score(r)
            if J < self.J:
                self.J, self.r = J, r
                # where the incumbent comes from: which kept iterate, which rounding
                print(f"  incumbent J {J:.6g} from iterate {self.n} "
                      f"({'top-x' if k == 0 else f'sample {k}'}, grey {grey:.3f})", flush=True)

    def consider(self, r: np.ndarray, label: str) -> None:
        """A 0/1 candidate from outside this incumbent's iterates (a coarse start's winner)."""
        key = np.packbits(r > 0).tobytes()
        if key in self._seen:
            return
        self._seen.add(key)
        J = self._score(r)
        if J < self.J:
            self.J, self.r = J, r
            print(f"  incumbent J {J:.6g} from {label}", flush=True)

    def best(self) -> tuple[float, np.ndarray]:
        """(exact J, clearing) of the best candidate."""
        return (self.J if self.scorer is None else self.rel.exact(self.r)), self.r

    def __call__(self, x: np.ndarray, stage: tuple[float, float]) -> None:
        self.n += 1
        if self.n % self.every == 0:
            self.offer(x, stage)

    def polish(self, x: np.ndarray, tries: int, width: int, everything: bool, pairs: bool,
               log=print) -> None:
        """Exchange refinement of the incumbent (`exchange`) over every building if `everything`,
        else over the undecided ones: grey in the final iterate x, or where the incumbent and x's
        top-x rounding disagree (late samples flip them; NOTES "Where .r2's winners come from").
        Ranked by the gradient at the 0/1 incumbent in the scorer's eps world (the greedy's
        tension)."""
        if self.scorer is None:
            raise ValueError("polish ranks with the eps-world scorer: give the plan .k<n>s<eps>")
        cost = self.rel.c.cost
        und = (np.arange(len(x)) if everything
               else np.flatnonzero(((x > 0.05) & (x < 0.95))
                                   | ((round_by_x(x, cost, self.budget, first=self.first) > 0)
                                      != (self.r > 0))))
        movable = und if self.first is None else und[self.first[und] == 0]   # held stay held
        J0 = self.J
        self.J, self.r, moves, used = exchange(
            self._score, lambda r: self.scorer.value_sgrad(r)[1], self.J, self.r, movable,
            cost, self.budget, tries, width, pairs)
        log(f"  polish: {len(movable)} {'buildings' if everything else 'undecided buildings'}, "
            f"{moves} moves kept, {used} scorings, J {J0:.6g} -> {self.J:.6g}")


def simp(rel: Relaxation, x0: np.ndarray, budget: float,
         stages: tuple[tuple[float, float], ...], iters: int, update: Update, log=print,
         xlo: np.ndarray | None = None, watch=None) -> np.ndarray:
    """SIMP continuation from x0: for (q, beta) in stages, up to `iters` updates by `update`
    (classic OC: x <- clip(x (-g / (lam c H'(x)))^1/2) within 0.2 of x) within [xlo (default
    XMIN), 1], lam bisected so that c . H(x) = budget (H = rel.proj; H' cancels the one in g, so
    the ratio is the sensitivity to the projected clearing per unit cost). Non-convex: a local
    method. `watch(x, (q, beta))`, if given, sees every iterate and its stage."""
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
        step = update.start()
        for it in range(iters):
            J, gs = rel.value_sgrad(x)
            B = np.maximum(-gs, 1e-300) / cost
            trial = step.trial(x, B, lo_)
            lo, hi = 1e-300, 1e300
            for _ in range(200):
                lam = np.sqrt(lo * hi)
                xn = trial(lam)
                lo, hi = (lam, hi) if cost @ rel.proj(xn) > budget else (lo, lam)
                if hi / lo < 1 + 1e-9:
                    break
            change = float(np.abs(xn - x).max())
            x = xn
            if watch is not None:
                watch(x, (q, beta))
            if it % 5 == 0 or change < 1e-3:
                sx = rel.proj(x)
                log(f"    simp q {q:g} beta {beta:g} it {it:2d} J {J:.6g} grey "
                    f"{np.mean((sx > 0.05) & (sx < 0.95)):.3f} change {change:.3f} "
                    f"{time.time() - t0:.0f}s")
            if change < 1e-3:
                break
    rel.q, rel.beta = 1.0, 0.0
    return x


def _clearing(bid: str, device: str, along: str, h: float = 0.5) -> Clearing:
    """`along`: the metric's conductance, or <metric>@<search>: SIMP's gradient under the search
    one (as the greedy's translucent search), every score under the metric."""
    [b] = common.build_blocks([bid])
    scans = lifted.scans_of(device)
    specs = along.split("@")
    p = lifted.Params(3.0, 8, along=lifted.along_of(specs[0], scans),
                      solver=lifted.solver_of(device))
    return Clearing(b, h, p, population=common.POPULATIONS["area"],
                    search=lifted.along_of(specs[1], scans) if len(specs) == 2 else None)


def one(bid: str, power: float, device: str, plan: Plan, along: str, budget: float) -> dict:
    c = _clearing(bid, device, along)
    rel = Relaxation(c, power, rtol=plan.rtol, eps=plan.eps)
    log = lambda s: print(f"{bid} {s}", flush=True)  # noqa: E731
    t0 = time.time()
    out = dict(block=bid, n=c.n, power=power, plan=plan.name, along=along)
    if plan.fw > 0:
        fw = frank_wolfe(rel, budget, plan.fw, log=log)
        x = fw["x"]
        r = round_by_x(x, c.cost, budget)
        frac = (x > 1e-6) & (x < 1 - 1e-6)
        out.update(iters=len(fw["hist"]), relaxed_perm=rel.perm(fw["J"]),
                   bound_perm=rel.perm(fw["lb"]), rounded_perm=rel.perm(rel.exact(r)),
                   rounded_D=float(c.cost @ r), frac_share=float(np.mean(frac)),
                   frac_cost=float(c.cost[frac].sum()), cleared=np.flatnonzero(r).tolist(),
                   x=x.tolist())
    else:
        x = start_x(0, c.cost, budget)
    starts = max(plan.starts, 1)
    if starts > 1 and (plan.fw > 0 or plan.coarse > 0):
        raise ValueError(f"multi-start is built for fw0 and without .c: {plan.name}")
    if plan.coarse_all > 0 and (plan.coarse > 0 or plan.fw > 0):
        raise ValueError(f".C is built for fw0 and without .c: {plan.name}")
    sample_in = plan.stages[-plan.late:] if plan.late > 0 else plan.stages
    inc = Incumbent(rel, budget, max(plan.keep, 1), samples=plan.samples, sample_in=sample_in,
                    score_eps=plan.kscore)
    watch = inc if plan.keep > 0 else None
    if plan.seed:
        picker = clear.picker_of(plan.seed, clear.sweep_of(device))
        order = [j for pk, _ in clear.grow(c, picker, power, budget, rel.J0) for j in pk.cleared]
        c.removed[:] = False
        inc.consider(cut_to_budget(order, c.cost, budget), f"the greedy {plan.seed}")
        log(f"  seeded by {plan.seed}: {time.time() - t0:.0f}s")
    xs = x
    if plan.coarse_all > 0:
        rel_c = Relaxation(_clearing(bid, device, along, plan.coarse_all), power,
                           rtol=plan.rtol, eps=plan.eps)
        log(f"  all stages at h {plan.coarse_all:g}: {rel_c.c.sc.grid.inside.sum()} cells, "
            f"fine {c.sc.grid.inside.sum()}; {starts} start(s)")
        for k in range(starts):
            inc_c = Incumbent(rel_c, budget, max(plan.keep, 1), samples=plan.samples,
                              sample_in=sample_in, score_eps=plan.kscore)
            xk = simp(rel_c, start_x(k, c.cost, budget), budget, plan.stages, plan.iters,
                      plan.update, log=log, watch=inc_c if plan.keep > 0 else None)
            inc_c.offer(xk, plan.stages[-1])
            J_before = inc.J
            inc.consider(inc_c.r, f"start {k}'s winner at h {plan.coarse_all:g}")
            if inc.J < J_before:
                xs = xk                                   # the polish's undecided set from it
            del inc_c
        del rel_c
        c.p.solver.release()     # the coarse systems' cached blocks: 1558 ran out of 32 GB without

    else:
        for k in range(starts):
            xk = x if k == 0 else start_x(k, c.cost, budget)
            stages = plan.stages
            if plan.coarse > 0:
                rel_c = Relaxation(_clearing(bid, device, along, plan.coarse), power,
                                   rtol=plan.rtol, eps=plan.eps)
                log(f"  coarse h {plan.coarse:g}: {rel_c.c.sc.grid.inside.sum()} cells, fine "
                    f"{c.sc.grid.inside.sum()}")
                xk = simp(rel_c, xk, budget, stages[:-1], plan.iters, plan.update, log=log,
                          watch=watch)
                del rel_c
                stages = stages[-1:]
            J_before = inc.J
            xk = simp(rel, xk, budget, stages, plan.iters, plan.update, log=log, watch=watch)
            inc.offer(xk, plan.stages[-1])                # the last iterate's rounding
            if k == 0 or inc.J < J_before:
                xs = xk
    if plan.polish > 0:
        for everything in plan.polish_sets:
            inc.polish(xs, plan.polish, plan.pwidth, everything, plan.polish_pairs, log=log)
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
    if (plan.coarse > 0 or plan.polish > 0 or plan.coarse_all > 0 or plan.starts > 1
            or plan.seed):
        raise ValueError(f"coarse grids, polish and multi-start are not built for path: "
                         f"{plan.name}")
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
        inc = Incumbent(rel, D, max(plan.keep, 1), samples=plan.samples,
                        sample_in=plan.stages[-plan.late:] if plan.late > 0 else plan.stages,
                        first=fixed.astype(float), score_eps=plan.kscore)
        x = simp(rel, x, D, plan.stages, plan.warm if warm else plan.iters, plan.update,
                 log=lambda s: None,
                 xlo=np.where(fixed, 1.0, XMIN), watch=inc if plan.keep > 0 else None)
        inc.offer(x, plan.stages[-1])
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


def rows_of(plan: Plan, along: str, budget: float) -> Path:
    return OUT / along / plan.name / f"D{budget:g}"


_CFG: dict = {}


def _run(bid: str) -> str | None:
    """`bid` if it failed (reported, and main exits non-zero at the end)."""
    mode, power, device, plan, along, budget = (_CFG[k] for k in (
        "mode", "power", "device", "plan", "along", "budget"))
    try:
        if mode == "one":
            pd.DataFrame([one(bid, power, device, plan, along, budget)]).to_parquet(
                rows_of(plan, along, budget) / f"{bid}_p{power:g}.parquet")
        else:
            out = rows_dir(f"SIMP{plan.name}", 0.5, "area", power, along) / f"{bid}.parquet"
            path(bid, power, device, plan, along).to_parquet(out)
            print(f"{time.strftime('%H:%M:%S')} {bid} path done", flush=True)
    except Exception as e:
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:300], flush=True)
        return bid
    return None


def _target(mode: str, bid: str, power: float, plan: Plan, along: str, budget: float) -> Path:
    if mode == "one":
        return rows_of(plan, along, budget) / f"{bid}_p{power:g}.parquet"
    return rows_dir(f"SIMP{plan.name}", 0.5, "area", power, along) / f"{bid}.parquet"


def main(mode: str, ids: list[str], power: float, device: str, plan: Plan, along: str,
         workers: int, budget: float) -> None:
    if mode not in ("one", "path"):
        raise ValueError(f"unknown mode {mode!r}")
    _target(mode, ids[0], power, plan, along, budget).parent.mkdir(parents=True, exist_ok=True)
    _CFG.update(mode=mode, power=power, device=device, plan=plan, along=along, budget=budget)
    todo = [i for i in ids if not _target(mode, i, power, plan, along, budget).exists()]
    if workers == 1:
        failed = [f for f in map(_run, todo) if f]
    else:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
            failed = [f for f in pool.imap_unordered(_run, todo) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


if __name__ == "__main__":
    ids = common.recipients() if sys.argv[2] == "all" else sys.argv[2].split(",")
    main(sys.argv[1], ids, float(sys.argv[3]), sys.argv[4], plan_of(sys.argv[5]), sys.argv[6],
         int(sys.argv[7]) if len(sys.argv) > 7 else 1,
         float(sys.argv[8]) if len(sys.argv) > 8 else D_LENS)
