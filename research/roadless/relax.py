"""The clearing problem as one convex program, solved by SIMP (and Frank-Wolfe for the bound).

Clear a fraction x_j in [0, 1] of each building: on the gap complex (gap_clear.Relaxation) a
building's passages conduct by k_j = eps + (1 - eps) proj(x_j)^q, each passage by the harmonic mean
of its triangles' factors, each triangle's the least k of the footprints whose shy zone holds it
(concave and non-decreasing), and by the Dirichlet principle P(w) = max_u 2 f^T u - sum_e w_e
(du_e)^2 is a maximum of functions affine and non-increasing in w. So P(x) is convex at q 1, and so
is min P(x) s.t. sum_j c_j x_j <= D, 0 <= x <= 1 (c_j the population share). J_p for p != 1 is
not convex; Frank-Wolfe is then a local method and the gap is not a bound.

Objective: J_p in the eps world (buildings as eps material, as the tension), so J_eps <= J
for every clearing and a lower bound on J_eps bounds every real clearing. Gradient: by the adjoint
(gap_clear.Model.grad; ties shared, a subgradient of J: the gap stays a valid bound). Linear step:
a fractional knapsack (the batch pickers' rule, best gain per unit population up to D). Step size:
a parabola through J(0), J'(0) = -gap and J(1), checked, and shorter steps when neither it nor the
full step descends (FW_BACKOFF). Rounding: clear buildings by decreasing x while they fit, then
score exactly.

    CUDA_PATH=/usr PYTHONPATH=. uv run python research/roadless/relax.py one|path <ids,|all> <p> <cpu|gpu> <plan> <metric> [workers] [budget]

metric: the gap complex's name (gap_clear.ComplexMetric; cpu or gpu, its factorization's),
resolved here (metrics.metric_of) and handed down. one: SIMP at one budget (default D_LENS, 0.05;
plus the Frank-Wolfe relaxation and bound if fw > 0), rows in relax_rows/<metric>/<plan>/D<budget>/.
path: nested budgets D 0.01 .. 0.15, rows in the greedy's format (clear.rows_dir, picker
SIMP<plan>).

Both free first (B's first stage, gap_clear.Model.free_first): the crossings that free the homes
walled in at baseline, cheapest per unit of population, within the budget, held at 1 through SIMP,
its roundings and the polish, which search with the rest of the budget; where no home is walled in
nothing is held. one's rows add free_D (the free stage's D), free_cleared (its buildings) and
enclosed (B (i) of the answer: the population share it leaves walled in,
gap_clear.Model.enclosed_left). fw, .m, .g and .G are not built for held buildings: a block whose
free stage holds any raises.

plan: fw<FW iterations, 0 = start SIMP from uniform x>.q<final q>.i<OC updates per q>, e.g.
fw0.q3.i10; the optional suffixes are Plan's (.e .b .k[s] .r[l] .p .w .m .g .G .u). (The raster's
coarse plans, `.c` and `.C`, its grids, `.h`, `.a` and `.g<d0>x<smax>...`, and its solve
tolerance `.t<rtol>` went with it: every solve on the complex is direct. The raster's rows keep
their names, `.t` and all: records, read by no code; NOTES, "The consolidated pass".)
"""
from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import search  # noqa: E402
from clear import EPS, rows_dir  # noqa: E402

if TYPE_CHECKING:
    import gap_clear    # gap_clear imports this module at run time

D_LENS = 0.05          # Lens A's budget (owner, 2026-10-03: 0.10 saturates under sightline)
OUT = HERE / "relax_rows"


def project(x: np.ndarray, beta: float) -> np.ndarray:
    """The clearing a design x stands for: x itself, or with beta > 0 a smooth step at ETA, H(x)
    = (tanh(beta ETA) + tanh(beta (x - ETA))) / (tanh(beta ETA) + tanh(beta (1 - ETA))), so a
    mostly-closed building conducts ~nothing (no grey leaks) and 0, 1 stay 0, 1. Conductance
    share H(x)^q, cost c . H(x). Every relaxation's (gap_clear.Relaxation.proj)."""
    if beta == 0.0:
        return x
    return ((np.tanh(beta * ETA) + np.tanh(beta * (x - ETA)))
            / (np.tanh(beta * ETA) + np.tanh(beta * (1 - ETA))))


def dproject(x: np.ndarray, beta: float) -> np.ndarray:
    if beta == 0.0:
        return np.ones_like(x)
    return (beta * (1 - np.tanh(beta * (x - ETA)) ** 2)
            / (np.tanh(beta * ETA) + np.tanh(beta * (1 - ETA))))


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
# frank_wolfe: when neither the parabola's step nor the full one descends, the step shrinks by
# FW_BACKOFF down to FW_STEP_FLOOR before it stops. J is nonsmooth (edge weights are minimums of
# open fractions, their gradient a subgradient shared among tied cells), and a direction from a
# subgradient can ascend at its first step yet descend at a shorter one: without backing off
# 41132 at eps 1e-4 stopped at iteration 8 with a third of J still in its gap. Every lower bound
# stays valid (any subgradient's), so going on only raises the bound and lowers J.
FW_BACKOFF = 4.0
FW_STEP_FLOOR = 1e-3


def frank_wolfe(rel: gap_clear.Relaxation, budget: float, iters: int, log=print) -> dict:
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
        if Jg >= J:                                     # neither descends: back off along d
            gam = min(gam, 1.0) / FW_BACKOFF
            while gam >= FW_STEP_FLOOR:
                Jg = rel.value(x + gam * d)
                if Jg < J:
                    log(f"    it {it:2d} backed off to step {gam:.3g}")
                    break
                gam /= FW_BACKOFF
        if Jg >= J:                                     # no progress at any step tried: stop
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
    optimality-criteria updates each, closed buildings conducting `eps` while optimizing (named only when not the tension's),
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
    `p`; `pP` the undecided then every one), with two-for-one moves if `polish_pairs` (`x2`),
    its candidates scored as `polish_scoring` says (search.PolishScoring: in the scorer's eps
    world, unnamed, or exactly, `E`; ranked in that eps world either way);
    `seed` (one only): that greedy picker's clearing within the budget offered to the incumbent
    before SIMP runs (`.g<picker>`: the polish starts from the better of the two); `track`
    (one only): that greedy's clearing polished as a second track, the better of the two
    polished results kept (`.G<picker>`: a better start can polish to a worse end -- 5706); with
    `track_gate` >= 0 (`.G<picker>e<pct>`) the greedy's track is polished only if its unpolished
    score is within that many percent of SIMP's polished one (replayed on the 13 and the 46, the
    greedy track won only where its raw clearing was already ahead);
    `warm`
    > 0 (path only): each budget after the first starts from the last budget's x (the free
    buildings lifted to the uniform share) with `warm` updates per stage; `starts` > 1 (one
    only): that many SIMP runs (start_x), one incumbent over all of them; `update` the design
    update rule (`.u<name>`; OC's classic rule is not named)."""
    fw: int
    qmax: float
    iters: int
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
    starts: int = 0
    seed: str = ""
    track: str = ""
    track_gate: float = -1.0
    update: Update = DEFAULT_UPDATE
    polish_scoring: search.PolishScoring = search.InEps()

    @property
    def name(self) -> str:
        return (f"fw{self.fw}.q{self.qmax:g}.i{self.iters}"
                + (f".e{self.eps:g}" if self.eps != EPS else "")
                + (f".b{self.b0:g}-{self.bmax:g}" if self.b0 > 0 else "")
                + (f".k{self.keep}" if self.keep > 0 else "")
                + (f"s{self.kscore:g}" if self.kscore > 0 else "")
                + (f".r{self.samples}" if self.samples > 0 else "")
                + (f"l{self.late}" if self.late > 0 else "")
                + (f".{''.join('P' if a else 'p' for a in self.polish_sets)}{self.polish}"
                   f"w{self.pwidth}" + ("x2" if self.polish_pairs else "")
                   + self.polish_scoring.suffix if self.polish > 0 else "")
                + (f".w{self.warm}" if self.warm > 0 else "")
                + (f".m{self.starts}" if self.starts > 1 else "")
                + (f".g{self.seed}" if self.seed else "")
                + (f".G{self.track}" if self.track else "")
                + (f"e{self.track_gate:g}" if self.track and self.track_gate >= 0 else "")
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


SIMP_DEFAULT = "fw0.q3.i10.e0.0001.k1s1e-06.p256w8x2"     # SIMP's default (plan_of's spec)


def plan_of(spec: str) -> Plan:
    m = re.fullmatch(r"fw(?P<fw>\d+)\.q(?P<q>[0-9.]+)\.i(?P<i>\d+)"
                     r"(?:\.e(?P<e>[0-9.e-]+?))?(?:\.b(?P<b0>[0-9.]+)-(?P<bmax>[0-9.]+))?"
                     r"(?:\.k(?P<k>\d+)(?:s(?P<ks>[0-9.e-]+))?)?(?:\.r(?P<r>\d+)(?:l(?P<rl>\d+))?)?"
                     r"(?:\.(?P<pset>p|P|pP)(?P<p>\d+)w(?P<pw>\d+)(?P<px>x2)?(?P<pe>E?))?"
                     r"(?:\.w(?P<w>\d+))?(?:\.m(?P<m>\d+))?"
                     r"(?:\.g(?P<g>S[0-9.]+cat(?:w\d+)?))?(?:\.G(?P<G>S[0-9.]+cat(?:w\d+)?)(?:e(?P<Ge>[0-9.]+))?)?"
                     r"(?:\.u(?:oc(?P<oc>[0-9.]+)m(?P<ocm>[0-9.]+)|mma(?P<mma>[0-9.]+)))?",
                     spec)
    if m is None:
        raise ValueError(f"unknown plan {spec!r}")
    g = m.groupdict()
    update: Update = (OC(float(g["oc"]), float(g["ocm"])) if g["oc"] is not None
                      else MMA(float(g["mma"])) if g["mma"] is not None
                      else DEFAULT_UPDATE)
    return Plan(fw=int(g["fw"]), qmax=float(g["q"]), iters=int(g["i"]),
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
                polish_scoring=search.polish_scoring_of(g["pe"] or ""),
                warm=0 if g["w"] is None else int(g["w"]),
                starts=0 if g["m"] is None else int(g["m"]), seed=g["g"] or "", track=g["G"] or "",
                track_gate=-1.0 if g["Ge"] is None else float(g["Ge"]),
                update=update)


class Incumbent:
    """The best 0/1 clearing among SIMP's iterates, each `every`-th rounded (`first` cleared
    before any other) and scored exactly. SIMP's iterates can hold a gate and lose it: on 22422
    the rounding of update 3 scores 0.810 and the last 0.012, the gate and its neighbours, being
    substitutes, flipping between x 0.3 and 0.5 until the continuation drops them (NOTES)."""

    def __init__(self, rel: gap_clear.Relaxation, budget: float, every: int, *, samples: int,
                 sample_in: tuple[tuple[float, float], ...],
                 first: np.ndarray | None = None, score_eps: float = 0.0):
        self.rel, self.budget, self.every, self.first = rel, budget, every, first
        # samples > 0: each offered iterate of a stage in `sample_in` ((q, beta)) is also
        # rounded that many times at random
        self.samples, self.sample_in = samples, sample_in
        self.rng = np.random.default_rng(SAMPLE_SEED)
        # score_eps > 0: candidates scored in the eps world at score_eps under the METRIC
        # conductance (rel may optimize under a search one), q 1, no projection
        self.scorer = rel.metric_world(score_eps) if score_eps > 0 else None
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
        """A 0/1 candidate from outside this incumbent's iterates (a greedy's clearing)."""
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

    def polish(self, x: np.ndarray, tries: int, width: int, everything: bool, pairs: bool, *,
               scoring: search.PolishScoring, log=print) -> None:
        """Exchange refinement of the incumbent (search.exchange_round) over every building if
        `everything`, else over the undecided ones: grey in the final iterate x, or where the
        incumbent and x's top-x rounding disagree (late samples flip them; NOTES "Where .r2's
        winners come from"). Ranked by the gradient at the 0/1 incumbent in the scorer's eps world
        (the greedy's tension), its candidates scored as `scoring` says (the incumbent rescored
        there first); its J then in that world (a second track's gate compares it with the
        greedy's raw J in the scorer's: exact and eps 1e-6 differ by about 1e-6 of J)."""
        if self.scorer is None:
            raise ValueError("polish ranks with the eps-world scorer: give the plan .k<n>s<eps>")
        cost = self.rel.c.cost
        und = (np.arange(len(x)) if everything
               else np.flatnonzero(((x > 0.05) & (x < 0.95))
                                   | ((round_by_x(x, cost, self.budget, first=self.first) > 0)
                                      != (self.r > 0))))
        movable = und if self.first is None else und[self.first[und] == 0]   # held stay held
        J0 = self.J
        c = self.rel.c
        c.removed[:] = self.r > 0
        s = search.SearchState(c=c, power=self.rel.power, score=search.UNSCORED, world=None,
                               order=[], scorings=0, movable=np.isin(np.arange(c.n), movable))
        moves = search.polish(scoring.world(self.scorer), budget=self.budget, width=width,
                              pairs=pairs, pool=PAIR_POOL, tries=tries)(s, search.Silent())
        self.J, self.r, used = s.score.J, c.removed.astype(float), s.scorings
        c.removed[:] = False                    # relax.one keeps its clearings in x, not here
        log(f"  polish: {len(movable)} {'buildings' if everything else 'undecided buildings'}, "
            f"{moves} moves kept, {used} scorings, J {J0:.6g} -> {self.J:.6g}")


def simp(rel: gap_clear.Relaxation, x0: np.ndarray, budget: float,
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


def one(bid: str, power: float, metric: gap_clear.ComplexMetric, plan: Plan, budget: float, *,
        demand: common.Demand) -> dict:
    """SIMP at one budget on `bid` under `metric`, J_power under `demand` (common.Demand: J_1 is
    P_q under a common.PriorityDemand), the cost the area population's."""
    [b] = common.build_blocks([bid])
    c = metric.clearing(b, demand=demand)
    rel = c.relaxation(power, q=1.0, eps=plan.eps)
    log = lambda s: print(f"{bid} {s}", flush=True)  # noqa: E731
    t0 = time.time()
    # B's first stage: the crossings that free the enclosed homes, held at 1 through SIMP (xlo),
    # the incumbent's roundings (first) and its polish (not movable); none enclosed, none held
    held = c.free_first(budget=budget, held=np.zeros(c.n, dtype=bool))
    if held.any() and (plan.fw > 0 or plan.starts > 1 or plan.seed or plan.track):
        raise NotImplementedError(f"{bid}: the free stage holds {int(held.sum())} buildings, and "
                                  f"fw, .m, .g and .G are not built for held buildings: "
                                  f"{plan.name}")
    xlo = np.where(held, 1.0, XMIN)
    out = dict(block=bid, n=c.n, power=power, demand=demand.name, plan=plan.name,
               along=metric.name, device=metric.device)
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
    elif held.any():
        # path's start with its fixed set: held at 1, the rest an even share of the budget left
        rest = budget - float(c.cost[held].sum())
        x = np.where(held, 1.0, rest / float(c.cost[~held].sum()))
    else:
        x = start_x(0, c.cost, budget)
    starts = max(plan.starts, 1)
    if starts > 1 and plan.fw > 0:
        raise ValueError(f"multi-start is built for fw0: {plan.name}")
    sample_in = plan.stages[-plan.late:] if plan.late > 0 else plan.stages
    inc = Incumbent(rel, budget, max(plan.keep, 1), samples=plan.samples, sample_in=sample_in,
                    first=held.astype(float), score_eps=plan.kscore)
    watch = inc if plan.keep > 0 else None
    if plan.seed and plan.track:
        raise ValueError(f"a greedy seeds the incumbent or runs as a second track, not both: "
                         f"{plan.name}")
    r_greedy = None
    if plan.seed or plan.track:
        spec = plan.seed or plan.track
        rnd = search.greedy_round(spec)
        s = search.SearchState.start(c, power=power, score=search.Score(*c.baseline(power)),
                                     world=search.EXACT)
        search.greedy(rnd, budget)(s, search.Silent())
        c.removed[:] = False
        r_greedy = cut_to_budget(s.order, c.cost, budget)
        if plan.seed:
            inc.consider(r_greedy, f"the greedy {plan.seed}")
        log(f"  the greedy {spec}: {time.time() - t0:.0f}s")
    xs = x
    for k in range(starts):
        xk = x if k == 0 else start_x(k, c.cost, budget)
        J_before = inc.J
        xk = simp(rel, xk, budget, plan.stages, plan.iters, plan.update, log=log, xlo=xlo,
                  watch=watch)
        inc.offer(xk, plan.stages[-1])                # the last iterate's rounding
        if k == 0 or inc.J < J_before:
            xs = xk
    if plan.polish > 0:
        for everything in plan.polish_sets:
            inc.polish(xs, plan.polish, plan.pwidth, everything, plan.polish_pairs,
                       scoring=plan.polish_scoring, log=log)
    if plan.track:
        inc_g = Incumbent(rel, budget, 1, samples=0, sample_in=(), score_eps=plan.kscore)
        inc_g.consider(r_greedy, f"the greedy {plan.track}, the second track")
        gated = plan.track_gate >= 0 and inc_g.J > inc.J * (1 + plan.track_gate / 100)
        if gated:
            log(f"  the greedy's track not polished: {inc_g.J:.6g} raw, SIMP {inc.J:.6g} polished")
        for everything in plan.polish_sets if plan.polish > 0 and not gated else ():
            inc_g.polish(xs, plan.polish, plan.pwidth, everything, plan.polish_pairs,
                         scoring=plan.polish_scoring, log=log)
        log(f"  tracks: SIMP {inc.J:.6g}, the greedy {inc_g.J:.6g}")
        if inc_g.J < inc.J:
            inc = inc_g
    Jbest, rs = inc.best()
    out.update(simp_perm=rel.perm(Jbest), simp_D=float(c.cost @ rs),
               simp_grey=float(np.mean((xs > 0.05) & (xs < 0.95))),
               simp_cleared=np.flatnonzero(rs).tolist(), simp_x=xs.tolist(),
               free_D=float(c.cost[held].sum()), free_cleared=np.flatnonzero(held).tolist(),
               enclosed=c.enclosed_left(rs > 0), t=time.time() - t0)
    print(f"{bid} n={c.n} p={power:g} {metric.name} {plan.name}: "
          + (f"relaxed {out['relaxed_perm']:.4f} bound {out['bound_perm']:.4f} rounded "
             f"{out['rounded_perm']:.4f}; " if plan.fw > 0 else "")
          + f"SIMP rounded {out['simp_perm']:.4f} (D {out['simp_D']:.4f}, grey "
          f"{out['simp_grey']:.3f}, enclosed {out['enclosed']:.4f}); {out['t']:.0f}s",
          flush=True)
    return out


PATH_DS = tuple(round(0.01 * k, 2) for k in range(1, 16))


def path(bid: str, power: float, metric: gap_clear.ComplexMetric, plan: Plan, *,
         demand: common.Demand) -> pd.DataFrame:
    """Nested SIMP: for D in PATH_DS, everything cleared so far held at 1, the rest reset to an
    even share of the remaining budget and the full q continuation rerun (warm-starting the
    rest at q = qmax traps them: a building SIMP has switched off, x = XMIN, has gradient
    ~ q x^(q-1) ~ 0 and multiplicative updates, so it never comes back -- measured: -0.076 Lens
    A against the greedy), round, score exactly. plan.warm > 0: after the first budget, the
    rest start from the last x lifted to at least that even share (so none is trapped at
    XMIN), and each stage gets plan.warm updates. Each budget frees first (Model.free_first
    within D, from the held set): its crossings join the held set before SIMP. Rows in the
    greedy's format: `cleared` = the buildings new at this step, the free stage's among them."""
    if plan.polish > 0 or plan.starts > 1 or plan.seed or plan.track:
        raise ValueError(f"polish, multi-start and greedy seeds are not built for path: "
                         f"{plan.name}")
    [b] = common.build_blocks([bid])
    c = metric.clearing(b, demand=demand)
    rel = c.relaxation(power, q=1.0, eps=plan.eps)
    J0, P0 = rel.J0, c.baseline(power)[1]
    rows = [dict(block=bid, n=c.n, step=0, D=0.0, perm=0.0, perm1=0.0, cleared=[], P0=P0,
                 t=0.0)]
    fixed = np.zeros(c.n, dtype=bool)
    x = np.full(c.n, PATH_DS[0])
    t0 = time.time()
    for i, D in enumerate(PATH_DS):
        last = fixed
        fixed = c.free_first(budget=D, held=last)
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
        new = np.flatnonzero(r & ~last)
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
    """`bid` if it failed (reported, and main exits non-zero at the end). The rows (named by
    the power alone) are the population's own demand's; pweight.py runs priority-weighted P."""
    mode, power, metric, plan, budget = (_CFG[k] for k in (
        "mode", "power", "metric", "plan", "budget"))
    demand = common.PopulationDemand()
    try:
        if mode == "one":
            pd.DataFrame([one(bid, power, metric, plan, budget, demand=demand)]).to_parquet(
                rows_of(plan, metric.name, budget) / f"{bid}_p{power:g}.parquet")
        else:
            out = rows_dir(f"SIMP{plan.name}", power=power, metric=metric.name) / f"{bid}.parquet"
            path(bid, power, metric, plan, demand=demand).assign(device=metric.device).to_parquet(
                out)
            print(f"{time.strftime('%H:%M:%S')} {bid} path done", flush=True)
    except Exception as e:
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:300], flush=True)
        return bid
    return None


def _target(mode: str, bid: str, power: float, plan: Plan, along: str, budget: float) -> Path:
    if mode == "one":
        return rows_of(plan, along, budget) / f"{bid}_p{power:g}.parquet"
    return rows_dir(f"SIMP{plan.name}", power=power, metric=along) / f"{bid}.parquet"


def main(mode: str, ids: list[str], power: float, metric: gap_clear.ComplexMetric, plan: Plan,
         workers: int, budget: float) -> None:
    if mode not in ("one", "path"):
        raise ValueError(f"unknown mode {mode!r}")
    _target(mode, ids[0], power, plan, metric.name, budget).parent.mkdir(parents=True,
                                                                         exist_ok=True)
    _CFG.update(mode=mode, power=power, metric=metric, plan=plan, budget=budget)
    todo = [i for i in ids if not _target(mode, i, power, plan, metric.name, budget).exists()]
    if workers == 1:
        failed = [f for f in map(_run, todo) if f]
    else:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
            failed = [f for f in pool.imap_unordered(_run, todo) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


if __name__ == "__main__":
    import metrics
    ids = common.recipients() if sys.argv[2] == "all" else common.ids_of(sys.argv[2])
    plan = plan_of(sys.argv[5])
    main(sys.argv[1], ids, float(sys.argv[3]),
         metrics.metric_of(sys.argv[6], device=sys.argv[4]), plan,
         int(sys.argv[7]) if len(sys.argv) > 7 else 1,
         float(sys.argv[8]) if len(sys.argv) > 8 else D_LENS)
