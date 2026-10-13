"""Guards of priority-weighted P (common.PriorityDemand, pweight.py), on real blocks, each with its
fault injection (a patch in one process, no file edited): a guard passes on the code as it is and
must fail under its fault.

    S=research/roadless/pweight_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S identity <ids,> [fault]
    uv run python $S total <ids,> [fault]
    uv run python $S displacement <ids,> [fault]
    uv run python $S gradient <ids,> [fault]

`identity`: P_0 is P exactly. PriorityDemand(0)'s injection is the population's, bit for bit, and
so are its field, its baseline's P0 and escape times, and P's and J_2's values on a random clearing
(every system seeded alike: pyamg's setup draws from numpy's global RNG, which otherwise moves a
solve by ~1e-11). Fault `no-population`: the injection a_i alone, the population dropped.

`total`: the injection's total is the population's for every q of Q_TOTAL: the injection summed
over the homes not enclosed against their population, and the field's total against the
population field's, within TOTAL_TOL relative. Fault `brief`: the brief's a_i = (u0_i /
mean_w(u0))^q, which keeps the total at q 0 and 1 only.

`displacement`: a clearing's displacement does not depend on the demand. Clearing.cost, the
population, B (i) at baseline and of a random clearing, and that clearing's area displacement under
P_{1/2}, P_1 and J_2 against P's, bit for bit. Fault `cost-from-demand`: Clearing.cost from the
injection.

`gradient`: relax.Relaxation's gradient of P_q (q 1/2 and 1) and of P at a random fractional
clearing, SIMP's q 1 and 3, in SIMP base's eps world (1e-4), against central differences (step
FD_STEP, solves to FD_RTOL from zero, so P is second-order accurate) on GRAD_BUILDINGS buildings:
the largest difference within GRAD_TOL of the largest |gradient| checked. Fault `J-population`:
the scorer's J weighting the population while the solve injects the priority demand (lam = u is
then not J's adjoint).
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import lifted  # noqa: E402
import pweight  # noqa: E402
import relax  # noqa: E402

Q_TOTAL = (0.5, 1.0, 2.0)
TOTAL_TOL = 1e-12
GRAD_BUILDINGS = 6
GRAD_SIMP_Q = (1.0, 3.0)
GRAD_EPS = 1e-4          # SIMP base's (pweight.PLAN.eps)
FD_STEP = 1e-4
FD_RTOL = 1e-9
GRAD_TOL = 1e-5
SEED = 20261012
POP = common.PopulationDemand()


def _clearing(bid: str, demand: common.Demand) -> clear.Clearing:
    return relax._clearing(bid, pweight.DEVICE, pweight.ALONG, pweight.PLAN.mesh, demand=demand)


def _random_clearing(c: clear.Clearing, rng: np.random.Generator) -> np.ndarray:
    """A random set of buildings within pweight.BUDGET (by Clearing.cost), in a random order."""
    x = np.zeros(c.n, dtype=bool)
    left = pweight.BUDGET
    for j in rng.permutation(c.n):
        if c.cost[j] <= left:
            x[j] = True
            left -= c.cost[j]
    return x


@contextmanager
def _seeded_systems() -> Iterator[None]:
    """Every lifted.System set up after numpy's global RNG is seeded SEED: the same field gives
    the same solve, bit for bit."""
    init = lifted.System.__init__

    def seeded(self, *a, **k):
        np.random.seed(SEED)
        init(self, *a, **k)
    lifted.System.__init__ = seeded
    try:
        yield
    finally:
        lifted.System.__init__ = init


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
        with _seeded_systems():
            cw, c0 = _clearing(bid, POP), _clearing(bid, common.PriorityDemand(0.0))
            x = _random_clearing(cw, np.random.default_rng(SEED))
            op = cw.open_(x)
            uw, u0 = cw.sc.home_u(op), c0.sc.home_u(op)
        same = dict(injection=np.array_equal(c0.sc.inj, cw.sc.w),
                    field=np.array_equal(c0.sc.f, cw.sc.f),
                    P0=c0.sc.P0 == cw.sc.P0,
                    u0=np.array_equal(c0.sc.u0, cw.sc.u0, equal_nan=True),
                    P=c0.sc.J(u0, 1.0) == cw.sc.J(uw, 1.0),
                    J2=c0.sc.J(u0, 2.0) == cw.sc.J(uw, 2.0))
        print(f"{bid}: q 0 against the population's demand, bit for bit: {same}; P0 "
              f"{c0.sc.P0:.12g} and {cw.sc.P0:.12g}", flush=True)
        if not all(same.values()):
            bad.append(bid)
    return bad


def total(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        cw = _clearing(bid, POP)
        live = ~cw.sc.enclosed
        for q in Q_TOTAL:
            c = _clearing(bid, common.PriorityDemand(q))
            rel_inj = abs(c.sc.inj[live].sum() / cw.sc.w[live].sum() - 1.0)
            rel_f = abs(c.sc.f.sum() / cw.sc.f.sum() - 1.0)
            a = c.sc.inj[live] / c.sc.w[live]
            ok = rel_inj <= TOTAL_TOL and rel_f <= TOTAL_TOL
            print(f"{bid} q {q:g}: injection's total off by {rel_inj:.1e}, field's {rel_f:.1e} "
                  f"(a from {a.min():.3f} to {a.max():.3f}) -> {'ok' if ok else 'FAILS'}",
                  flush=True)
            if not ok:
                bad.append(f"{bid} q {q:g}")
    return bad


def displacement(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        cp = _clearing(bid, POP)
        x = _random_clearing(cp, np.random.default_rng(SEED))
        ref = (cp.cost, cp.sc.w, cp.sc.enclosed0, cp.sc.enclosed_left(cp.open_(x)),
               float(cp.cost @ x))
        for o in pweight.OBJECTIVES:
            c = _clearing(bid, o.demand)
            got = (c.cost, c.sc.w, c.sc.enclosed0, c.sc.enclosed_left(c.open_(x)),
                   float(c.cost @ x))
            same = [np.array_equal(a, b) for a, b in zip(got, ref, strict=True)]
            print(f"{bid} {o.name}: cost, population, B (i) at baseline ({got[2]:.4f}) and of "
                  f"a random clearing ({got[3]:.4f}), its displacement ({got[4]:.4f}): {same}",
                  flush=True)
            if not all(same):
                bad.append(f"{bid} {o.name}")
    return bad


def _value(rel: relax.Relaxation, x: np.ndarray) -> float:
    rel._u = None                       # from zero: P second-order accurate in the solve
    return rel.value(x)


def gradient(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        for token in ("Pq0.5", "Pq1", "P"):
            o = pweight.objective_of(token)
            c = _clearing(bid, o.demand)
            rng = np.random.default_rng(SEED)
            x0 = rng.uniform(0.1, 0.4, c.n)
            pick = rng.choice(c.n, GRAD_BUILDINGS, replace=False)
            for sq in GRAD_SIMP_Q:
                rel = relax.Relaxation(c, o.power, q=sq, rtol=FD_RTOL, eps=GRAD_EPS)
                rel._u = None
                _, g = rel.value_grad(x0)
                fd = []
                for j in pick:
                    e = np.zeros(c.n)
                    e[j] = FD_STEP
                    fd.append((_value(rel, x0 + e) - _value(rel, x0 - e)) / (2 * FD_STEP))
                err = float(np.abs(np.array(fd) - g[pick]).max() / np.abs(g[pick]).max())
                ok = err <= GRAD_TOL
                print(f"{bid} {o.name} SIMP q {sq:g}: gradient against central differences on "
                      f"{GRAD_BUILDINGS} buildings, {err:.1e} of the largest -> "
                      f"{'ok' if ok else 'FAILS'}", flush=True)
                if not ok:
                    bad.append(f"{bid} {o.name} q {sq:g}")
    return bad


def _no_population(self, w, u0, enclosed):
    live = ~enclosed
    uq = np.where(live, np.nan_to_num(u0), 1.0) ** self.q
    return np.where(live, uq / float((w[live] * uq[live]).sum() / w[live].sum()), w)


def _brief(self, w, u0, enclosed):
    live = ~enclosed
    mean = float((w[live] * np.nan_to_num(u0)[live]).sum() / w[live].sum())
    return np.where(live, w * (np.where(live, np.nan_to_num(u0), 1.0) / mean) ** self.q, w)


def _cost_from_demand() -> Callable:
    init = clear.Clearing.__init__

    def patched(self, *a, **k):
        init(self, *a, **k)
        self.cost = self.sc.inj / self.sc.inj.sum()
    return patched


def _J_population(self, u, power):
    ok = ~self.enclosed
    return float((self.w[ok] * u[ok] ** power).sum())


GUARDS = {"identity": identity, "total": total, "displacement": displacement,
          "gradient": gradient}
FAULTS = {
    "identity": {"no-population": lambda: _patched(common.PriorityDemand, "injection",
                                                   _no_population)},
    "total": {"brief": lambda: _patched(common.PriorityDemand, "injection", _brief)},
    "displacement": {"cost-from-demand": lambda: _patched(clear.Clearing, "__init__",
                                                          _cost_from_demand())},
    "gradient": {"J-population": lambda: _patched(common.Scorer, "J", _J_population)},
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
