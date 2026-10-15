"""Guards of priority-weighted P (common.PriorityDemand, pweight.py) on the runtime's metric, the
gap complex (phase 2b: the guards ported from the raster's scorer to gap_clear.Model), on real
blocks, each with its fault injection (a patch in one process, no file edited): a guard passes on
the code as it is and must fail under its fault.

    S=research/roadless/pweight_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S identity <ids,> [fault]
    uv run python $S total <ids,> [fault]
    uv run python $S displacement <ids,> [fault]
    uv run python $S gradient <ids,> [fault]
    uv run python $S weights <ids,> [fault]

`identity`: P_0 is P exactly. PriorityDemand(0)'s injection is the population's, bit for bit, and
so are the model's baseline P and escape times under it, and P's and J_2's values on a random
clearing (the complex's solves are direct: the same system gives the same solve). Fault
`no-population`: the injection a_i alone, the population dropped.

`total`: the injection's total is the population's for every q of Q_TOTAL: the injection summed
over the homes with demand against their population, and the demand's points' total (every point,
the band's among them: the right-hand side drops those escaping at once) against the
population's, within TOTAL_TOL relative. Fault `brief`: the brief's a_i = (u0_i / mean_w(u0))^q,
which keeps the total at q 0 and 1 only.

`displacement`: a clearing's displacement does not depend on the demand. The model's cost, the
population, B (i) at baseline and of a random clearing, and that clearing's area displacement under
P_{1/2}, P_1 and J_2 against P's, bit for bit. Fault `cost-from-demand`: the model's cost from the
injection.

`gradient`: the relaxation's gradient of P_q (q 1/2 and 1) and of P at a random fractional
clearing, SIMP's q 1 and 3, in SIMP base's eps world (1e-4), against the closest of the central
differences at GRAD_STEPS on GRAD_BUILDINGS buildings: the largest difference within GRAD_TOL of
the largest |gradient| checked. Fault `J-population`: the model's J weighting the population while
the solve injects the priority demand (lam = u is then not J's adjoint).

`weights` (phase 2b): P_{1/2}'s weight of every home (gap_clear.Model.inj_q over the population)
against a direct computation from the baseline's escape times, solved afresh (the baseline's
operator assembled from the passages, gap_clear_check._operator, and solved directly; each home's
escape time its points' injection-weighted mean potential): a_i = u0_i^(1/2) / mean_w(u0^(1/2))
over the homes with demand, within WEIGHT_TOL relative. Fault `swap-time`: one home's weight made
from another's escape time (the most populous home takes the least populous one's).
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import pweight  # noqa: E402

Q_TOTAL = (0.5, 1.0, 2.0)
TOTAL_TOL = 1e-12
GRAD_BUILDINGS = 6
GRAD_SIMP_Q = (1.0, 3.0)
GRAD_EPS = 1e-4          # SIMP base's (pweight.PLAN.eps)
GRAD_STEPS = (1e-3, 1e-4, 1e-5)   # the closest central difference: the factors are piecewise
                                  # linear (one step straddling a kink of the minimum, another
                                  # not), and the solves' roundoff on P (9e4 on 20269) reaches
                                  # 1e-4 of the gradient at a step of 1e-5 and 1e-3 at 1e-6
                                  # (measured), where 1e-3 and 1e-4 agree within 6e-6
GRAD_TOL = 1e-4          # as gap_clear_check's gradient guard
WEIGHT_TOL = 1e-9        # the model's weights against the direct computation (a refined solve
                         # against an unrefined direct one: ~1e-12)
SEED = 20261012
POP = common.PopulationDemand()


def _model(bid: str, demand: common.Demand) -> gcl.Model:
    [b] = common.build_blocks([bid])
    return pweight.METRIC.clearing(b, demand=demand)


def _random_clearing(m: gcl.Model, rng: np.random.Generator) -> np.ndarray:
    """A random set of buildings within pweight.BUDGET (by the model's cost), in a random order."""
    x = np.zeros(m.n, dtype=bool)
    left = pweight.BUDGET
    for j in rng.permutation(m.n):
        if m.cost[j] <= left:
            x[j] = True
            left -= m.cost[j]
    return x


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def identity(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        mw, m0 = _model(bid, POP), _model(bid, common.PriorityDemand(0.0))
        x = _random_clearing(mw, np.random.default_rng(SEED))
        same = dict(injection=np.array_equal(m0.inj, mw.pop), P0=m0.P0 == mw.P0,
                    u0=np.array_equal(m0.u0, mw.u0, equal_nan=True),
                    P=m0.exact(x, 1.0) == mw.exact(x, 1.0), J2=m0.exact(x, 2.0) == mw.exact(x, 2.0))
        print(f"{bid}: q 0 against the population's demand, bit for bit: {same}; P0 "
              f"{m0.P0:.12g} and {mw.P0:.12g}", flush=True)
        if not all(same.values()):
            bad.append(bid)
    return bad


def total(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        mw = _model(bid, POP)
        live = ~mw.nodem
        for q in Q_TOTAL:
            m = _model(bid, common.PriorityDemand(q))
            rel_inj = abs(m.inj[live].sum() / mw.pop[live].sum() - 1.0)
            rel_f = abs(float((m.dem.S @ m.inj).sum()) / float((mw.dem.S @ mw.pop).sum()) - 1.0)
            a = m.inj[live] / m.pop[live]
            ok = rel_inj <= TOTAL_TOL and rel_f <= TOTAL_TOL
            print(f"{bid} q {q:g}: injection's total off by {rel_inj:.1e}, the demand points' "
                  f"{rel_f:.1e} (a from {a.min():.3f} to {a.max():.3f}) -> "
                  f"{'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} q {q:g}")
    return bad


def displacement(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        mp = _model(bid, POP)
        x = _random_clearing(mp, np.random.default_rng(SEED))
        ref = (mp.cost, mp.pop, float(mp.cost[mp.walled0].sum()), mp.enclosed_left(x),
               float(mp.cost @ x))
        for o in pweight.OBJECTIVES:
            m = _model(bid, o.demand)
            got = (m.cost, m.pop, float(m.cost[m.walled0].sum()), m.enclosed_left(x),
                   float(m.cost @ x))
            same = [np.array_equal(a, b) for a, b in zip(got, ref, strict=True)]
            print(f"{bid} {o.name}: cost, population, B (i) at baseline ({got[2]:.4f}) and of "
                  f"a random clearing ({got[3]:.4f}), its displacement ({got[4]:.4f}): {same}",
                  flush=True)
            if not all(same):
                bad.append(f"{bid} {o.name}")
    return bad


def gradient(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        for token in ("Pq0.5", "Pq1", "P"):
            o = pweight.objective_of(token)
            m = _model(bid, o.demand)
            rng = np.random.default_rng(SEED)
            x0 = rng.uniform(0.1, 0.4, m.n)
            pick = rng.choice(m.n, GRAD_BUILDINGS, replace=False)
            for sq in GRAD_SIMP_Q:
                rel = m.relaxation(o.power, q=sq, rtol=0.0, eps=GRAD_EPS)
                _, g = rel.value_grad(x0)
                errs = []
                for j in pick:
                    best = np.inf
                    for h in GRAD_STEPS:
                        e = np.zeros(m.n)
                        e[j] = h
                        fd = (rel.value(x0 + e) - rel.value(x0 - e)) / (2 * h)
                        best = min(best, abs(fd - g[j]))
                    errs.append(best)
                err = float(np.max(errs) / np.abs(g[pick]).max())
                ok = err <= GRAD_TOL
                print(f"{bid} {o.name} SIMP q {sq:g}: gradient against central differences on "
                      f"{GRAD_BUILDINGS} buildings, {err:.1e} of the largest -> "
                      f"{'ok' if ok else 'FAILS'}", flush=True)
                if not ok:
                    bad.append(f"{bid} {o.name} q {sq:g}")
    return bad


def weights(ids: list[str]) -> list[str]:
    from scipy.sparse.linalg import spsolve

    import gap_clear_check
    bad = []
    for bid in ids:
        m = _model(bid, POP)
        st = m.state(np.zeros(m.n))
        A, cells = gap_clear_check._operator(st)
        rhs = m._rhs(st, m.pop)
        u = np.zeros(st.sys.n)
        u[cells] = spsolve(A.tocsc(), rhs[cells])
        u0 = np.asarray(st.sys.PcT @ u)
        live = ~m.nodem
        q = gcl.PQ.q
        uq = u0[live] ** q
        a = uq / ((m.pop[live] * uq).sum() / m.pop[live].sum())
        got = (m.inj_q / m.pop)[live]
        err = float(np.max(np.abs(got / a - 1.0)))
        ok = err <= WEIGHT_TOL
        print(f"{bid}: P_{{{q:g}}}'s weights of {int(live.sum())} homes against the direct "
              f"computation from the baseline's escape times, within {err:.1e} (a from "
              f"{a.min():.3f} to {a.max():.3f}) -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _no_population(self, w, u0, enclosed):
    live = ~enclosed
    uq = np.where(live, np.nan_to_num(u0), 1.0) ** self.q
    return np.where(live, uq / float((w[live] * uq[live]).sum() / w[live].sum()), w)


def _brief(self, w, u0, enclosed):
    live = ~enclosed
    mean = float((w[live] * np.nan_to_num(u0)[live]).sum() / w[live].sum())
    return np.where(live, w * (np.where(live, np.nan_to_num(u0), 1.0) / mean) ** self.q, w)


_INJECTION = common.PriorityDemand.injection


def _swap_time(self, w, u0, enclosed):
    """PriorityDemand.injection with the most populous live home's escape time read from the
    least populous one's (fault `swap-time`)."""
    live = np.flatnonzero(~enclosed)
    hi, lo = live[np.argmax(w[live])], live[np.argmin(w[live])]
    u = np.array(u0, dtype=np.float64)
    u[hi] = u0[lo]
    return _INJECTION(self, w, u, enclosed)


def _cost_from_demand() -> Callable:
    init = gcl.Model.__init__

    def patched(self, *a, **k):
        init(self, *a, **k)
        self.cost_ = self.inj / self.inj.sum()
    return patched


def _J_population(self, u_home, inj, power):
    ok = ~self.nodem
    return float((self.pop[ok] * u_home[ok] ** power).sum())


GUARDS = {"identity": identity, "total": total, "displacement": displacement,
          "gradient": gradient, "weights": weights}
FAULTS = {
    "identity": {"no-population": lambda: _patched(common.PriorityDemand, "injection",
                                                   _no_population)},
    "total": {"brief": lambda: _patched(common.PriorityDemand, "injection", _brief)},
    "displacement": {"cost-from-demand": lambda: _patched(gcl.Model, "__init__",
                                                          _cost_from_demand())},
    "gradient": {"J-population": lambda: _patched(gcl.Model, "J", _J_population)},
    "weights": {"swap-time": lambda: _patched(common.PriorityDemand, "injection", _swap_time)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    if fault is None:
        bad = GUARDS[guard](ids)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard](ids)
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (2, 3) or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    main(a[0], a[1].split(","), a[2] if len(a) == 3 else None)
